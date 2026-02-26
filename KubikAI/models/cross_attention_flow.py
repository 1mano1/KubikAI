from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from KubikAI.models.base_flow_model import TimestepEmbedder, BaseFlowModel
from KubikAI.modules.transformer.modulated import ModulatedTransformerCrossBlock
from KubikAI.modules.spatial import patchify, unpatchify
from KubikAI.modules.transformer.blocks import AbsolutePositionEmbedder


class CrossAttentionFlowModel(BaseFlowModel):
    """
    A flow model that uses cross-attention to condition on a sequence of image features.
    """
    def __init__(
        self,
        image_feature_dim: int,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        # We need to re-initialize the blocks to use the correct context dimension
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                self.model_channels,
                # The context is now the image feature dimension
                ctx_channels=image_feature_dim, 
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                use_checkpoint=self.use_checkpoint,
                use_rope=(self.pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(self.num_blocks)
        ])
        
        # Re-initialize weights
        self.initialize_weights()
        if self.use_fp16:
            self.convert_to_fp16()


    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, cond_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention.

        Args:
            x (torch.Tensor): Input latent, shape (B, C, H, W, D)
            t (torch.Tensor): Timestep embedding, shape (B,)
            cond (torch.Tensor): Global condition embedding (e.g., from DINOv2 CLS token), shape (B, cond_channels)
            cond_features (torch.Tensor): Sequence of image features for cross-attention, shape (B, L, image_feature_dim)
        """
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3]

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        if self.pe_mode == 'ape' and hasattr(self, 'pos_emb'):
             h = h + self.pos_emb[None]
        
        # We still use the global `cond` for the time modulation, as in the original model
        t_emb = self.t_embedder(t) + cond 
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond_features = cond_features.type(self.dtype)

        for block in self.blocks:
            # Here is the key change: we pass cond_features as the context
            h = block(h, t_emb, cond_features) 

        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h
