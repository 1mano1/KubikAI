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
        # Using trimesh.proximity.signed_distance
        # positive outside, negative inside
        sdf = trimesh.proximity.signed_distance(mesh, points)
        
        # Our model expects: negative outside, positive inside (typical in some SDF implementations)
        # Or standard: positive outside, negative inside. 
        # Let's stick to standard and adjust in dataset if needed.
        
        return points.astype(np.float32), sdf.astype(np.float32)
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")
        return None, None

def process_single_model(args):
    mesh_path, output_dir = args
    model_id = os.path.basename(os.path.dirname(mesh_path))
    if not model_id or len(model_id) < 5: # fallback for flat structures
        model_id = os.path.basename(mesh_path).split('.')[0]
        
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
