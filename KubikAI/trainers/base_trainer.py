import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class BaseTrainer:
    """
    A simplified, single-GPU, non-distributed trainer.
    """
    def __init__(self, models, dataset, batch_size_per_gpu, max_steps, optimizer, i_log, i_save, output_dir, **kwargs):
        self.models = models
        self.dataset = dataset
        self.batch_size = batch_size_per_gpu
        self.max_steps = max_steps
        self.optimizer_config = optimizer
        self.i_log = i_log
        self.i_save = i_save
        self.output_dir = output_dir
        self.step = 0

        # Move models to GPU
        for name, model in self.models.items():
            self.models[name] = model.cuda()

        self.init_dataloader()
        self.init_optimizer()
        
        # Automatic Resume
        self.resume()

        print("--- Simplified BaseTrainer Initialized ---")
        print(f"  - Models: {list(self.models.keys())}")
        print(f"  - Dataset size: {len(self.dataset)}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Starting at Step: {self.step}")
        print(f"  - Max steps: {self.max_steps}")
        print("----------------------------------------")

    def init_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.data_iter = iter(self.dataloader)

    def init_optimizer(self):
        model_params = sum([list(model.parameters()) for model in self.models.values()], [])
        self.optimizer = getattr(torch.optim, self.optimizer_config['name'])(model_params, **self.optimizer_config['args'])

    def resume(self):
        import glob
        ckpt_dir = os.path.join(self.output_dir, 'ckpts')
        if not os.path.exists(ckpt_dir):
            return
        misc_files = sorted(glob.glob(os.path.join(ckpt_dir, 'misc_step*.pt')))
        if not misc_files:
            return
        latest_misc = misc_files[-1]
        print(f"Found existing checkpoint: {latest_misc}")
        misc_ckpt = torch.load(latest_misc, map_location='cuda')
        self.step = misc_ckpt['step']
        self.optimizer.load_state_dict(misc_ckpt['optimizer'])
        for name in self.models.keys():
            model_path = os.path.join(ckpt_dir, f'{name}_step{self.step:07d}.pt')
            if os.path.exists(model_path):
                print(f"Loading model weight: {model_path}")
                self.models[name].load_state_dict(torch.load(model_path, map_location='cuda'))

    def get_next_batch(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda(non_blocking=True)
        return batch

    def training_losses(self, data):
        raise NotImplementedError

    def run_step(self, data):
        self.optimizer.zero_grad()
        loss, status = self.training_losses(**data)
        loss['loss'].backward()
        self.optimizer.step()
        log = {'loss': {k: v.item() for k, v in loss.items()}}
        log['status'] = {k: v.item() for k, v in status.items()}
        return log

    def save(self):
        print(f'\nSaving checkpoint at step {self.step}...', end='')
        checkpoints_dir = os.path.join(self.output_dir, 'ckpts')
        os.makedirs(checkpoints_dir, exist_ok=True)
        for name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, f'{name}_step{self.step:07d}.pt'))
        misc_ckpt = {'optimizer': self.optimizer.state_dict(), 'step': self.step}
        torch.save(misc_ckpt, os.path.join(checkpoints_dir, f'misc_step{self.step:07d}.pt'))
        print(' Done.')

    def run(self):
        print(f"Training loop starting. Current step: {self.step}, Max steps: {self.max_steps}")
        while self.step < self.max_steps:
            batch = self.get_next_batch()
            log = self.run_step(batch)
            self.step += 1
            
            if self.step % self.i_log == 0:
                log_str = f"Step {self.step:07d} "
                for k, v in log['loss'].items():
                    log_str += f"| {k}: {v:.4f} "
                print(log_str) # Force print for background logs

            if self.step > 0 and self.step % self.i_save == 0:
                self.save()
        
        print("Training finished successfully.")
        self.save()
