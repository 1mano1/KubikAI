from typing import *
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

from KubikAI.trainers.base_trainer import BaseTrainer

class SdfVaeTrainer(BaseTrainer):
    """
    Trainer for the Signed Distance Function VAE.
    """
    def __init__(
        self,
        models,
        dataset,
        lambda_kl,
        lambda_recon,
        **kwargs
    ):
        # Pass general arguments to the BaseTrainer
        super().__init__(models=models, dataset=dataset, **kwargs)
        
        self.lambda_kl = lambda_kl
        self.lambda_recon = lambda_recon
        
        # In our new `train_vae.py`, the model is passed under the key 'vae'
        assert 'vae' in self.models, "SdfVaeTrainer expects a model named 'vae'"

    def training_losses(
        self,
        points: torch.Tensor,
        sdf: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses for the SdfVAE.

        Args:
            points: The [B, N, 3] tensor of sample points.
            sdf: The [B, N, 1] tensor of ground truth signed distances.

        Returns:
            A dict of losses and a dict of statuses.
        """
        # We can now directly access the model
        vae_model = self.models['vae']
        
        # Forward pass
        sdf_pred, mean, logvar = vae_model(points, sdf)
        
        # Calculate losses
        terms = edict()
        
        # 1. Reconstruction Loss (L1 loss on the SDF values)
        recon_loss = F.l1_loss(sdf_pred, sdf)
        terms["recon_loss"] = recon_loss * self.lambda_recon

        # 2. KL Divergence Loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        terms["kl_loss"] = kl_loss * self.lambda_kl

        # Total loss
        terms["loss"] = terms["recon_loss"] + terms["kl_loss"]

        # Status dict for logging
        status = {
            'loss': terms.loss, # Keep as tensor for backward()
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
        }

        return terms, status
