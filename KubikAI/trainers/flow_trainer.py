from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from easydict import EasyDict as edict

from KubikAI.trainers.base_trainer import BaseTrainer

# This new FlowTrainer combines the logic from the original project's:
# - FlowMatchingTrainer
# - ClassifierFreeGuidanceMixin
# - ImageConditionedMixin
# - our own CrossAttentionTrainer logic
# into a single, self-contained class that inherits from our simplified BaseTrainer.

class FlowTrainer(BaseTrainer):
    def __init__(
        self,
        models,
        dataset,
        # FlowMatching args
        t_schedule: dict,
        sigma_min: float,
        # CFG arg
        p_uncond: float,
        # ImageConditioning arg
        image_cond_model: str,
        # Our CrossAttention args
        lambda_mse: float,
        **kwargs
    ):
        super().__init__(models=models, dataset=dataset, **kwargs)
        
        # Store args
        self.t_schedule = t_schedule
        self.sigma_min = sigma_min
        self.p_uncond = p_uncond
        self.image_cond_model_name = image_cond_model
        self.lambda_mse = lambda_mse

        # Lazy-load the image conditioning model
        self.image_cond_model = None
        
        assert 'flow' in self.models, "FlowTrainer expects a model named 'flow'"

    def _init_image_cond_model(self):
        """
        Initialize the image conditioning model (DINOv2).
        """
        print("Initializing DINOv2 image conditioning model...")
        dinov2_model = torch.hub.load('facebookresearch/dinov2', self.image_cond_model_name, pretrained=True)
        dinov2_model.eval().cuda()
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model = {
            'model': dinov2_model,
            'transform': transform,
        }
        print("DINOv2 model initialized.")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode the image using the DINOv2 model.
        """
        if self.image_cond_model is None:
            self._init_image_cond_model()
        
        image = self.image_cond_model['transform'](image)
        features = self.image_cond_model['model'](image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    # --- Core Flow Matching Methods ---

    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
        x_t = (1 - t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise
        return x_t

    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (1 - self.sigma_min) * noise - x_0

    def sample_t(self, batch_size: int) -> torch.Tensor:
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size, device='cuda')
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size, device='cuda') * std + mean)
        else:
            raise ValueError(f"Unknown t_schedule: {self.t_schedule['name']}")
        return t

    # --- Main Training Logic ---

    def training_losses(
        self,
        latent: torch.Tensor, # This is x_0, the latent vector from the VAE
        image: torch.Tensor, # This is the conditioning image
        **kwargs
    ) -> Tuple[Dict, Dict]:
        
        # 1. Encode the conditioning image
        all_cond_features = self.encode_image(image) # (B, L, C)
        global_cond = all_cond_features[:, 0]       # (B, C) - CLS token
        local_cond_features = all_cond_features[:, 1:] # (B, L-1, C) - Patch tokens

        # 2. Classifier-Free Guidance Logic
        # Create masks for dropping conditions
        is_dropped = (torch.rand(global_cond.shape[0], device='cuda') < self.p_uncond)
        
        # Create null conditions
        null_global_cond = torch.zeros_like(global_cond)
        null_local_features = torch.zeros_like(local_cond_features)
        
        # Select final conditions based on mask
        final_global_cond = torch.where(is_dropped.view(-1, 1), null_global_cond, global_cond)
        final_local_features = torch.where(is_dropped.view(-1, 1, 1), null_local_features, local_cond_features)

        # 3. Standard Flow Matching steps
        noise = torch.randn_like(latent)
        t = self.sample_t(latent.shape[0])
        x_t = self.diffuse(latent, t, noise=noise)
        
        # 4. Forward pass through the model
        flow_model = self.models['flow']
        pred = flow_model(
            x_t, 
            t * 1000, 
            cond=final_global_cond,
            cond_features=final_local_features
        )
        
        # 5. Calculate loss
        target = self.get_v(latent, noise)
        terms = edict()
        terms["mse"] = F.mse_loss(pred, target)
        terms["loss"] = terms["mse"] * self.lambda_mse

        status = { 'loss': terms.loss }
        return terms, status
