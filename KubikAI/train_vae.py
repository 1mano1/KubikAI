import os
import sys
import json
import argparse
import importlib
import torch
from easydict import EasyDict as edict

# Add project root to path to allow absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def import_class(name: str):
    """
    Import a class from a string path.
    e.g., 'KubikAI.models.SdfVAE' -> <class 'KubikAI.models.SdfVAE'>
    """
    module_name, class_name = name.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except ImportError as e:
        print(f"Could not import {name}: {e}")
        raise

def main(cfg):
    """
    Main training loop for the VAE.
    """
    print("--- VAE Training Script ---")
    
    # 1. Load Dataset
    print("Loading dataset...")
    dataset_class = import_class(cfg.dataset.name)
    data_roots = [path.strip() for path in cfg.data_dir.split(',')]
    dataset = dataset_class(data_roots, **cfg.dataset.args)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 2. Build Model
    print("Building model...")
    # The config for the VAE only has one model, so we can access it directly.
    vae_model_cfg = list(cfg.models.values())[0]
    model_class = import_class(vae_model_cfg.name)
    model = model_class(**vae_model_cfg.args).cuda()
    
    # We wrap the model in a dictionary as the trainer expects it.
    model_dict = {'vae': model}
    print("Model built successfully.")

    # 3. Build Trainer
    print("Building trainer...")
    trainer_class = import_class(cfg.trainer.name)
    # Pass load_dir and resume_step to the trainer if provided
    trainer_kwargs = dict(cfg.trainer.args)
    if cfg.get('load_dir'):
        trainer_kwargs['load_dir'] = cfg.load_dir
    if cfg.get('resume_step'):
        trainer_kwargs['resume_step'] = cfg.resume_step
        
    trainer = trainer_class(model_dict, dataset, **trainer_kwargs, output_dir=cfg.output_dir)
    print("Trainer built successfully.")

    # 4. Start Training
    print("\nStarting training...")
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dedicated training script for the SDF VAE model.")
    parser.add_argument('--config', type=str, required=True, help='Experiment config file (e.g., KubikAI/configs/kubikai_sdf_vae_v1.json)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints and logs.')
    parser.add_argument('--data_dir', type=str, required=True, help='Comma-separated paths to dataset directories.')
    parser.add_argument('--load_dir', type=str, default=None, help='Directory containing the checkpoint to resume from.')
    parser.add_argument('--resume_step', type=int, default=None, help='Step number to resume training from.')
    
    args = parser.parse_args()
    
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Combine args and config
    cfg = edict()
    cfg.update(vars(args))
    cfg.update(config)

    # Prepare output directory
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("\n--- Configuration ---")
    print(json.dumps(cfg, indent=4))
    print("---------------------\n")
    
    main(cfg)
