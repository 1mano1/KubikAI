import os
import glob
import numpy as np
import trimesh
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def mesh_to_sdf_samples(mesh_path, num_samples=32768):
    """
    Converts a mesh to SDF samples (points and distances).
    """
    try:
        # Load mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # 1. Normalize to unit sphere/box
        center = mesh.centroid
        mesh.apply_translation(-center)
        scale = 1.0 / np.max(mesh.extents)
        mesh.apply_scale(scale)

        # 2. Sample points
        # - Near surface points (High precision for details)
        # - Uniform points (For general shape)
        n_surface = num_samples // 2
        n_uniform = num_samples - n_surface

        # Sample points near the surface
        points_surface = mesh.sample(n_surface)
        points_surface += np.random.normal(scale=0.01, size=points_surface.shape)

        # Sample points uniformly in the bounding box
        points_uniform = np.random.uniform(-0.5, 0.5, (n_uniform, 3))

        points = np.concatenate([points_surface, points_uniform], axis=0)

        # 3. Calculate SDF (Signed Distance Function)
        # Using proximity.closest_point is MUCH faster than proximity.signed_distance
        # because it only calculates magnitude, not the complex sign math.
        closest_points, distances, _ = trimesh.proximity.closest_point(mesh, points)
        
        # To get the sign (inside = negative, outside = positive), we use ray casting.
        # We shoot a ray from the point in a random direction. If it hits an ODD number of faces, 
        # it's inside. If EVEN, it's outside. (Classic point-in-polygon algorithm)
        # trimesh handles this via `contains` which is optimized.
        
        # mesh.contains expects waterproof meshes. If the mesh has holes, it might fail.
        # But it's 100x faster than proximity.signed_distance.
        try:
            is_inside = mesh.contains(points)
            # Apply sign: negative if inside, positive if outside
            sdf = np.where(is_inside, -distances, distances)
        except Exception:
            # Fallback for non-watertight meshes (a bit slower but safer)
            sdf = trimesh.proximity.signed_distance(mesh, points)
            # trimesh standard: positive outside, negative inside.
            # But the fallback signed_distance function returns negative inside.
            pass

        # Reshape to (N, 1)
        sdf = sdf.reshape(-1, 1)
        
        return points.astype(np.float32), sdf.astype(np.float32)
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None, None

def process_single_model(args):
    mesh_path, output_dir = args
    # Create a truly unique ID by combining the parent directory name and the filename
    # e.g., "folder123_mesh.npz" to avoid collisions in flat or nested datasets
    parent_dir = os.path.basename(os.path.dirname(mesh_path))
    file_name = os.path.basename(mesh_path).split('.')[0]
    
    # If the parent dir is just 'models' or something generic, add a small hash of the path
    import hashlib
    path_hash = hashlib.md5(mesh_path.encode('utf-8')).hexdigest()[:6]
    
    model_id = f"{parent_dir}_{file_name}_{path_hash}"
    
    save_path = os.path.join(output_dir, f"{model_id}.npz")
    
    if os.path.exists(save_path):
        return True

    points, sdf = mesh_to_sdf_samples(mesh_path)
    
    if points is not None:
        np.savez_compressed(save_path, points=points, sdf=sdf)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Batch process Plyverse meshes to SDF for KubikAI.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to Plyverse part (e.g. /kaggle/input/.../plyverse_part1)")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save .npz files")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of models to process in this batch")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of parallel workers")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Search for .ply files in the nested structure
    print(f"Searching for models in {args.input_dir}...")
    all_meshes = glob.glob(os.path.join(args.input_dir, "**", "*.ply"), recursive=True)
    
    if not all_meshes:
        # Try .obj just in case
        all_meshes = glob.glob(os.path.join(args.input_dir, "**", "*.obj"), recursive=True)

    print(f"Found {len(all_meshes)} models. Processing first {args.limit}...")
    all_meshes = all_meshes[:args.limit]

    # Prepare arguments for multiprocessing
    tasks = [(path, args.output_dir) for path in all_meshes]

    with Pool(args.workers) as p:
        results = list(tqdm(p.imap(process_single_model, tasks), total=len(tasks), desc="Converting to SDF"))

    success_count = sum(results)
    print(f"\nFinished! Successfully processed {success_count} models.")
    print(f"Files saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
