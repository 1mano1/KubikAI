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
    Main training loop for the Flow model.
    """
    print("--- Flow Model Training Script ---")
    
    # 1. Load Dataset
    print("Loading dataset...")
    dataset_class = import_class(cfg.dataset.name)
    
    # Update dataset args with command line paths if provided
    dataset_args = dict(cfg.dataset.args)
    dataset_args['encoded_root'] = cfg.encoded_dir
    dataset_args['processed_root'] = cfg.processed_dir
    
    dataset = dataset_class(**dataset_args)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 2. Build Model
    print("Building model...")
    flow_model_cfg = list(cfg.models.values())[0]
    model_class = import_class(flow_model_cfg.name)
    model = model_class(**flow_model_cfg.args).cuda()
    # The trainer expects a dictionary of models
    model_dict = {'flow': model}
    print("Model built successfully.")

    # 3. Build Trainer
    print("Building trainer...")
    trainer_class = import_class(cfg.trainer.name)
    trainer = trainer_class(models=model_dict, dataset=dataset, **cfg.trainer.args, output_dir=cfg.output_dir)
    print("Trainer built successfully.")

    # 4. Start Training
    print("\nStarting training...")
    trainer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dedicated training script for the Cross-Attention Flow model.")
    parser.add_argument('--config', type=str, required=True, help='Experiment config file (e.g., KubikAI/configs/kubikai_flow_v1.json)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save checkpoints and logs.')
    parser.add_argument('--encoded_dir', type=str, required=True, help='Path to the directory with encoded latent vectors.')
    parser.add_argument('--processed_dir', type=str, required=True, help='Path to the processed datasets directory (containing the original render images).')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    cfg = edict()
    cfg.update(vars(args))
    cfg.update(config)

    os.makedirs(cfg.output_dir, exist_ok=True)
    
    print("\n--- Configuration ---")
    print(json.dumps(cfg, indent=4))
    print("---------------------\n")
    
    main(cfg)
