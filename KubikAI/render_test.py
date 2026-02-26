import os
import sys
import trimesh
import numpy as np
from PIL import Image

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def render_mesh(obj_path, output_png):
    import pyrender
    
    # Load mesh
    mesh = trimesh.load(obj_path)
    
    # Setup scene
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[1.0, 1.0, 1.0, 1.0])
    
    # Add mesh to scene
    py_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(py_mesh)
    
    # Setup camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    
    # Position camera: look at the center of the mesh
    center = mesh.centroid
    scale = np.max(mesh.extents)
    
    # Camera pose
    camera_pose = np.array([
        [1.0,  0.0,  0.0, center[0]],
        [0.0,  0.8, -0.6, center[1] - scale * 1.5],
        [0.0,  0.6,  0.8, center[2] + scale * 1.5],
        [0.0,  0.0,  0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)
    
    # Add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(light, pose=camera_pose)
    
    # Render
    r = pyrender.OffscreenRenderer(512, 512)
    color, _ = r.render(scene)
    r.delete()
    
    # Save
    img = Image.fromarray(color)
    img.save(output_png)
    print(f"Render saved to {output_png}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    try:
        render_mesh(args.input, args.output)
    except Exception as e:
        print(f"Error rendering: {e}")
