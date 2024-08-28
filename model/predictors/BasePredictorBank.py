from dataclasses import dataclass, field, asdict
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
from model import networks
from model.geometry.dmtet import DMTetGeometry
from model.utils import misc
from .BasePredictorBase import netDINOConfig, BasePredictorBase


@dataclass
class DMTetEmbConfig:
    grid_res: int = 64
    spatial_scale: float = 5.
    num_layers: int = 5
    hidden_size: int = 64
    embedder_freq: int = 8
    embed_concat_pts: bool = True
    init_sdf: Union[int, float, str] = None
    jitter_grid: float = 0.
    symmetrize: bool = False
    grid_res_coarse_epoch_range: List[int] = None
    grid_res_coarse: int = 128
    condition_choice: str = "mod"
    condition_dim: str = 128
    grid_res_coarse_iter_range: List[int] = None


@dataclass
class MemoryBankConfig:
    memory_bank_size: int = 60
    memory_bank_topk: int = 10
    memory_bank_dim: int = 128
    memory_bank_keys_dim: int = 384


@dataclass
class BasePredictorBankConfig:
    cfg_shape: DMTetEmbConfig
    cfg_dino: netDINOConfig
    cfg_bank: MemoryBankConfig


class BasePredictorBank(BasePredictorBase):
    def __init__(self, cfg: BasePredictorBankConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, BasePredictorBankConfig)
        del self.netShape
        self.netShape = DMTetGeometry(**asdict(self.cfg_shape))

        del self.netDINO
        dino_minmax_tensor = torch.FloatTensor(self.cfg_dino.minmax).repeat(self.cfg_dino.feature_dim, 1)  # Nx2
        embedder_scalar = 2 * np.pi / self.cfg_shape.spatial_scale * 0.9  # originally (-0.5, 0.5) * spatial_scale rescale to (-pi, pi) * 0.9
        self.netDINO = networks.CoordMLP(
            3,  # x, y, z coordinates
            self.cfg_dino.feature_dim,
            self.cfg_dino.num_layers,
            nf=self.cfg_dino.hidden_size,
            dropout=0,
            activation=self.cfg_dino.activation,
            min_max=dino_minmax_tensor,
            n_harmonic_functions=self.cfg_dino.embedder_freq,
            embedder_scalar=embedder_scalar,
            embed_concat_pts=self.cfg_dino.embed_concat_pts,
            extra_feat_dim=self.cfg_bank.memory_bank_dim,
            symmetrize=self.cfg_dino.symmetrize
        )

        assert self.cfg_bank.memory_bank_topk <= self.cfg_bank.memory_bank_size
        # self.memory_bank = torch.nn.Parameter(torch.nn.init.uniform_(torch.empty(self.cfg_bank.memory_bank_size, self.cfg_bank.memory_bank_dim), a=-0.05, b=0.05), requires_grad=True)
        
        # an empirical bank initialization
        bank_size = self.cfg_bank.memory_bank_size
        memory_bank = torch.nn.init.uniform_(torch.empty(7, self.cfg_bank.memory_bank_dim), a=-0.05, b=0.05)
        num_piece = bank_size // 7
        num_left = bank_size - num_piece * 7
        tmp_1 = torch.empty_like(memory_bank)
        tmp_1 = tmp_1.copy_(memory_bank)
        tmp_1 = tmp_1.unsqueeze(0).repeat(num_piece, 1, 1)
        tmp_1 = tmp_1.reshape(tmp_1.shape[0] * tmp_1.shape[1], tmp_1.shape[-1])
        tmp_2 = torch.empty_like(memory_bank)
        tmp_2 = tmp_2.copy_(memory_bank)
        tmp_2 = tmp_2[:num_left]
        tmp = torch.cat([tmp_1, tmp_2], dim=0)
        self.memory_bank = torch.nn.Parameter(tmp, requires_grad=True)
        
        memory_bank_keys = torch.nn.init.uniform_(torch.empty(self.cfg_bank.memory_bank_size, self.cfg_bank.memory_bank_keys_dim), a=-0.05, b=0.05)
        self.memory_bank_keys = torch.nn.Parameter(memory_bank_keys, requires_grad=True)


    def forward(self, total_iter=None, is_training=True, batch=None, bank_enc=None):
        images = batch[0]
        batch_size, num_frames, _, h0, w0 = images.shape
        images = images.reshape(batch_size*num_frames, *images.shape[2:])  # 0~1

        images_in = images * 2 - 1  # rescale to (-1, 1)
        batch_features = self.forward_frozen_ViT(images_in, bank_enc)
        batch_embedding, embeddings, weights = self.retrieve_memory_bank(batch_features, batch)
        bank_embedding_model_input = [batch_embedding, embeddings, weights]

        prior_shape = self.netShape.getMesh(total_iter=total_iter, jitter_grid=is_training, feats=batch_embedding)
        return prior_shape, self.netDINO, bank_embedding_model_input
    

    def forward_frozen_ViT(self, images, dino_enc):
        # this part use the frozen pre-train ViT
        x = images
        with torch.no_grad():
            b, c, h, w = x.shape
            dino_enc._feats = []
            dino_enc._register_hooks([11], 'key')
            #self._register_hooks([11], 'token')
            x = dino_enc.ViT.prepare_tokens(x)
            #x = self.ViT.prepare_tokens_with_masks(x)
            
            for blk in dino_enc.ViT.blocks:
                x = blk(x)
            out = dino_enc.ViT.norm(x)
            dino_enc._unregister_hooks()

            ph, pw = h // dino_enc.patch_size, w // dino_enc.patch_size
            patch_out = out[:, 1:]  # first is class token
            patch_out = patch_out.reshape(b, ph, pw, dino_enc.vit_feat_dim).permute(0, 3, 1, 2)

            patch_key = dino_enc._feats[0][:,:,1:]  # B, num_heads, num_patches, dim
            patch_key = patch_key.permute(0, 1, 3, 2).reshape(b, dino_enc.vit_feat_dim, ph, pw)

            global_feat = out[:, 0]
        
        return global_feat
    

    def retrieve_memory_bank(self, batch_features, batch):
        batch_size = batch_features.shape[0]
        
        query = torch.nn.functional.normalize(batch_features.unsqueeze(1), dim=-1)      # [B, 1, d_k]
        key = torch.nn.functional.normalize(self.memory_bank_keys, dim=-1)              # [size, d_k]
        key = key.transpose(1, 0).unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)             # [B, d_k, size]

        cos_dist = torch.bmm(query, key).squeeze(1)         # [B, size], larger the more similar
        rank_idx = torch.sort(cos_dist, dim=-1, descending=True)[1][:, :self.cfg_bank.memory_bank_topk] # [B, k]
        value = self.memory_bank.unsqueeze(0).repeat(batch_size, 1, 1).to(query.device)                         # [B, size, d_v]

        out = torch.gather(value, dim=1, index=rank_idx[..., None].repeat(1, 1, self.cfg_bank.memory_bank_dim))  # [B, k, d_v]

        weights = torch.gather(cos_dist, dim=-1, index=rank_idx)    # [B, k]
        weights = torch.nn.functional.normalize(weights, p=1.0, dim=-1).unsqueeze(-1).repeat(1, 1, self.cfg_bank.memory_bank_dim)    # [B, k, d_v] weights have been normalized

        out = weights * out
        out = torch.sum(out, dim=1)
        
        batch_mean_out = torch.mean(out, dim=0)

        weight_aux = {
            'weights': weights[:, :, 0], # [B, k], weights from large to small
            'pick_idx': rank_idx, # [B, k]
        }

        return batch_mean_out, out, weight_aux