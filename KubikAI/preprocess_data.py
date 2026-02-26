import os
import json
import argparse
import numpy as np
import trimesh
import subprocess
import logging
import math
from glob import glob
from PIL import Image
import scipy.ndimage
from scipy.interpolate import RegularGridInterpolator

# ===============================
# LOW DISCREPANCY SEQUENCE UTILS
# ===============================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[i], n) for i in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u = (u + offset[0]) % 1.0
    v = (v + offset[1]) % 1.0
    
    # Improved mapping to avoid singularities at poles
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

# ===============================
# PRE-PROCESSING FUNCTIONS
# ===============================

def render_views_with_pyrender(args, obj_path, output_dir):
    """
    Renders multiple views of a 3D model using PyRender.
    This is an alternative to the Blender pipeline.
    """
    logging.info(f"Rendering {args.num_views} views for {obj_path} with pyrender...")
    import pyrender

    # --- 1. Load and Normalize Mesh ---
    scene_or_mesh = trimesh.load(obj_path, force='mesh', process=True)
    
    # Normalize scene to fit into a unit box
    bounds = scene_or_mesh.bounds
    if bounds is None:
        raise ValueError("Could not determine mesh bounds.")
    
    center = scene_or_mesh.centroid
    scene_or_mesh.apply_translation(-center)
    
    scale = 1.0 / np.max(scene_or_mesh.extents)
    scene_or_mesh.apply_scale(scale)
    
    # --- 2. Setup Pyrender Scene ---
    pyrender_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0, 0, 0, 0])
    
    if isinstance(scene_or_mesh, trimesh.Scene):
        for geom in scene_or_mesh.geometry.values():
            # materials are handled by trimesh's visual properties
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
            pyrender_scene.add(mesh)
    else: # It's a single Trimesh geometry
        mesh = pyrender.Mesh.from_trimesh(scene_or_mesh, smooth=True)
        pyrender_scene.add(mesh)

    # --- 3. Setup Camera and Lighting ---
    fov = 40 / 180 * np.pi
    cam = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    
    # --- 4. Render Views ---
    r = pyrender.OffscreenRenderer(args.resolution, args.resolution)
    
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "frames": []
    }
    
    offset = (np.random.rand(), np.random.rand())
    for i in range(args.num_views):
        yaw, pitch = sphere_hammersley_sequence(i, args.num_views, offset)
        radius = np.random.uniform(1.8, 2.2)

        cam_x = radius * np.cos(yaw) * np.cos(pitch)
        cam_y = radius * np.sin(yaw) * np.cos(pitch)
        cam_z = radius * np.sin(pitch)

        camera_location = np.array([cam_x, cam_y, cam_z])
        
        # World-to-camera matrix
        # This is equivalent to look-at calculation
        forward = -camera_location / np.linalg.norm(camera_location)
        right = np.cross([0, 0, 1], forward)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross([0, 1, 0], forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = right
        cam_pose[:3, 1] = up
        cam_pose[:3, 2] = forward
        cam_pose[:3, 3] = camera_location
        
        pyrender_scene.add(cam, pose=cam_pose)
        pyrender_scene.add(light, pose=cam_pose)

        color, _ = r.render(pyrender_scene, flags=pyrender.RenderFlags.RGBA)

        img = Image.fromarray(color, 'RGBA')
        img_path = os.path.join(output_dir, f'{i:03d}.png')
        img.save(img_path)

        # Store metadata (transform_matrix is camera-to-world, so we need the inverse)
        transform_matrix = np.linalg.inv(cam_pose)
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": fov,
            "transform_matrix": transform_matrix.tolist()
        }
        to_export["frames"].append(metadata)
        
        for node in pyrender_scene.get_nodes():
            if node.camera or node.light:
                pyrender_scene.remove_node(node)
        
    r.delete()
    
    with open(os.path.join(output_dir, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
    # --- 5. Save Normalized Mesh ---
    normalized_mesh_path = os.path.join(output_dir, 'mesh.ply')
    scene_or_mesh.export(normalized_mesh_path)
    logging.info(f"Saved normalized mesh to {normalized_mesh_path}")


def generate_sdf(obj_path, output_path, num_points=250000):
    """
    Generates an SDF npz file from a 3D mesh using a robust, memory-stable
    voxelization and distance transform approach.
    """
    logging.info(f"Generating SDF for {obj_path} using voxelization and distance transform...")

    try:
        mesh = trimesh.load(obj_path, force='mesh')
        logging.info(f"Loaded mesh '{obj_path}' with {len(mesh.faces)} faces.")
    except Exception as e:
        logging.error(f"Failed to load mesh {obj_path}: {e}")
        raise

    # --- Create a Voxel Grid ---
    # Convert the mesh to a watertight-ish representation
    mesh.fill_holes()
    
    # Determine a suitable resolution for our grid
    resolution = 128 
    
    # Voxelize the mesh to a grid of the chosen resolution
    # The pitch is set to ensure the largest dimension fits within `resolution` voxels.
    pitch = mesh.extents.max() / (resolution - 1)
    voxels = mesh.voxelized(pitch=pitch)
    
    # The `matrix` attribute is a boolean grid of filled voxels
    voxel_matrix = voxels.matrix

    # --- Compute SDF using Distance Transform ---
    logging.info("Computing Signed Distance Field from voxel grid...")
    # Compute distance from any point to the nearest 'True' (inside)
    dist_inside = scipy.ndimage.distance_transform_edt(voxel_matrix)
    # Compute distance from any point to the nearest 'False' (outside)
    dist_outside = scipy.ndimage.distance_transform_edt(~voxel_matrix)
    
    # The SDF is the difference between these two distances
    sdf_grid = dist_outside - dist_inside

    # --- Create an interpolator for the SDF grid ---
    # This allows us to query the SDF at any point, not just on the grid
    grid_points = [np.arange(s) for s in sdf_grid.shape]
    interpolator = RegularGridInterpolator(grid_points, sdf_grid, bounds_error=False, fill_value=0)

    # --- Generate query points ---
    # We generate points in the original, unscaled mesh space
    original_mesh = trimesh.load(obj_path, force='mesh')
    
    num_uniform_points = int(num_points * 0.5)
    # Generate points within the bounding box of the original mesh
    uniform_points = np.random.uniform(original_mesh.bounds[0], original_mesh.bounds[1], size=(num_uniform_points, 3)).astype(np.float32)

    num_surface_points = num_points - num_uniform_points
    surface_points, _ = trimesh.sample.sample_surface(original_mesh, num_surface_points)
    surface_points += np.random.normal(scale=np.mean(original_mesh.extents) * 0.005, size=surface_points.shape)
    surface_points = surface_points.astype(np.float32)

    points = np.concatenate([uniform_points, surface_points], axis=0)

    # --- Map query points to voxel grid and interpolate ---
    # The voxel grid has its own coordinate system, we need to transform our points
    points_in_voxel_coords = trimesh.transform_points(points, np.linalg.inv(voxels.transform))
    
    logging.info("Interpolating SDF values for query points...")
    signed_distance = interpolator(points_in_voxel_coords)
    
    # The distance is in voxel units, convert it back to world units
    signed_distance *= pitch
    signed_distance = signed_distance.astype(np.float32)

    np.savez(output_path, points=points, sdf=signed_distance)
    logging.info(f"SDF data saved to {output_path}")




def main(args):
    """
    Main function to orchestrate the pre-processing pipeline.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(os.path.dirname(__file__), 'preprocess.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting pre-processing pipeline.")
    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output directory: {args.output_dir}")

    obj_files = glob(os.path.join(args.input_dir, '**', '*.obj'), recursive=True)
    if not obj_files:
        logging.warning(f"No .obj files found in {args.input_dir}. Exiting.")
        return

    logging.info(f"Found {len(obj_files)} models to process.")
    
    successful_models = []
    failed_models = []

    for obj_path in obj_files:
        model_name = os.path.splitext(os.path.basename(obj_path))[0]
        model_output_dir = os.path.join(args.output_dir, model_name)
        
        logging.info(f"----- Processing Model: {model_name} -----")
        
        try:
            sdf_output_dir = os.path.join(model_output_dir, 'sdf')
            renders_output_dir = os.path.join(model_output_dir, 'renders')
            os.makedirs(sdf_output_dir, exist_ok=True)
            os.makedirs(renders_output_dir, exist_ok=True)
            
            # --- 1. Render Views ---
            render_views_with_pyrender(args, obj_path, renders_output_dir)
            
            # --- 2. Generate SDF ---
            normalized_mesh_path = os.path.join(renders_output_dir, 'mesh.ply')
            sdf_output_path = os.path.join(sdf_output_dir, 'samples.npz')
            
            if os.path.exists(normalized_mesh_path):
                generate_sdf(normalized_mesh_path, sdf_output_path)
            else:
                raise FileNotFoundError(f"Normalized mesh not found for {model_name}")

            successful_models.append(model_name)
            logging.info(f"Successfully processed model: {model_name}")

        except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
            logging.error(f"Failed to process model {model_name}.")
            if isinstance(e, subprocess.CalledProcessError):
                logging.error(f"Blender failed with exit code {e.returncode}.")
                logging.error(f"Blender stdout: {e.stdout}")
                logging.error(f"Blender stderr: {e.stderr}")
            else:
                logging.error(f"Error details: {e}")
            failed_models.append(model_name)
            continue

    logging.info("\n----- Pre-processing Summary -----")
    logging.info(f"Successfully processed {len(successful_models)} models: {successful_models}")
    if failed_models:
        logging.warning(f"Failed to process {len(failed_models)} models: {failed_models}")
    logging.info("Pre-processing pipeline finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive pre-processing pipeline for 3D models for KubikAI.')
    
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the 3D model files (e.g., .obj).')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory where the processed datasets will be saved.')
    
    parser.add_argument('--num_views', type=int, default=50, help='Number of views to render per model.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the rendered images.')
    
    args = parser.parse_args()
    
    main(args)
