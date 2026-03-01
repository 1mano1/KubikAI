import os
import glob
import json
import argparse
from tqdm import tqdm
import numpy as np
import trimesh

def validate_sdf_dataset(data_dir, output_json, min_samples=16384):
    """
    Scans a massive Kaggle dataset of pre-computed SDFs (.npz).
    Ensures they are not corrupted and have enough samples.
    """
    print(f"Scanning for SDF files in: {data_dir}")
    sdf_files = glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)
    
    valid_files = []
    corrupted_files = []
    insufficient_samples = []

    for f in tqdm(sdf_files, desc="Validating SDFs"):
        try:
            data = np.load(f)
            points = data['points']
            sdf = data['sdf']
            
            # Check basic integrity
            if points.shape[0] != sdf.shape[0]:
                corrupted_files.append(f)
                continue
            
            # Check if it has enough samples for our 16k requirement
            if points.shape[0] < min_samples:
                insufficient_samples.append(f)
                continue
                
            # If everything is fine, keep it
            valid_files.append(f)
            
        except Exception as e:
            corrupted_files.append(f)

    # Save the manifest of good files
    manifest = {
        "dataset_dir": data_dir,
        "total_scanned": len(sdf_files),
        "valid_count": len(valid_files),
        "corrupted_count": len(corrupted_files),
        "insufficient_samples_count": len(insufficient_samples),
        "valid_files": valid_files
    }
    
    with open(output_json, 'w') as out_f:
        json.dump(manifest, out_f, indent=4)
        
    print(f"
--- Validation Complete ---")
    print(f"Total Scanned: {len(sdf_files)}")
    print(f"Valid for Training: {len(valid_files)}")
    print(f"Corrupted: {len(corrupted_files)}")
    print(f"Too Few Samples (<{min_samples}): {len(insufficient_samples)}")
    print(f"Manifest saved to: {output_json}")

def validate_raw_meshes(data_dir, output_json, min_faces=1000, max_faces=500000):
    """
    Scans a massive Kaggle dataset of raw meshes (.obj, .glb).
    Filters out models that are too low-poly (no detail) or too high-poly (OOM crashes).
    """
    print(f"Scanning for Mesh files in: {data_dir}")
    mesh_files = []
    for ext in ['*.obj', '*.glb', '*.gltf', '*.fbx', '*.ply']:
        mesh_files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    
    valid_files = []
    corrupted_files = []
    filtered_out = []

    for f in tqdm(mesh_files, desc="Validating Meshes"):
        try:
            # We use process=False to load faster, just checking metadata
            mesh = trimesh.load(f, force='mesh', process=False)
            
            # Check poly count
            if len(mesh.faces) < min_faces or len(mesh.faces) > max_faces:
                filtered_out.append(f)
                continue
                
            # Check if it's watertight (optional, but good for SDF)
            # if not mesh.is_watertight:
            #     filtered_out.append(f)
            #     continue

            valid_files.append(f)
        except Exception as e:
            corrupted_files.append(f)

    manifest = {
        "dataset_dir": data_dir,
        "total_scanned": len(mesh_files),
        "valid_count": len(valid_files),
        "filtered_out_count": len(filtered_out),
        "corrupted_count": len(corrupted_files),
        "valid_files": valid_files
    }
    
    with open(output_json, 'w') as out_f:
        json.dump(manifest, out_f, indent=4)
        
    print(f"
--- Validation Complete ---")
    print(f"Total Scanned: {len(mesh_files)}")
    print(f"Valid for Pipeline: {len(valid_files)}")
    print(f"Corrupted/Unreadable: {len(corrupted_files)}")
    print(f"Filtered (Poly count limits): {len(filtered_out)}")
    print(f"Manifest saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate massive Kaggle datasets before training.")
    parser.add_argument("--dir", type=str, required=True, help="Directory to scan (e.g., /kaggle/input/shapenet)")
    parser.add_argument("--type", type=str, choices=["sdf", "mesh"], required=True, help="Type of data to validate")
    parser.add_argument("--output", type=str, default="dataset_manifest.json", help="Output JSON manifest file")
    
    args = parser.parse_args()
    
    if args.type == "sdf":
        validate_sdf_dataset(args.dir, args.output)
    else:
        validate_raw_meshes(args.dir, args.output)
