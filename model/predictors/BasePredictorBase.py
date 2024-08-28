from dataclasses import dataclass, field, asdict
from typing import List, Union
import numpy as np
import torch
import torch.nn as nn
from model import networks
from model.geometry.dmtet import DMTetGeometry
from model.utils import misc


@dataclass
class DMTetConfig:
    grid_res: int = 64
    spatial_scale: float = 5.
    num_layers: int = 5
    hidden_size: int = 64
    embedder_freq: int = 8
    embed_concat_pts: bool = True
    init_sdf: Union[int, float, str] = None
    jitter_grid: float = 0.
    symmetrize: bool = False
    grid_res_coarse_iter_range: List[int] = None
    grid_res_coarse: int = 128


@dataclass
class netDINOConfig:
    feature_dim: int = 64
    num_layers: int = 5
    hidden_size: int = 64
    activation: str = "sigmoid"
    embedder_freq: int = 8
    embed_concat_pts: bool = True
    symmetrize: bool = False
    minmax: List[float] = field(default_factory=lambda: [0., 1.])


@dataclass
class BasePredictorConfig:
    cfg_shape: DMTetConfig
    cfg_dino: netDINOConfig


class BasePredictorBase(nn.Module):
    def __init__(self, cfg: BasePredictorConfig):
        super().__init__()
        misc.load_cfg(self, cfg, BasePredictorConfig)
        
        self.netShape = DMTetGeometry(**asdict(self.cfg_shape))

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
            extra_feat_dim=0,
            symmetrize=self.cfg_dino.symmetrize
        )
        
    def forward(self, total_iter=None, is_training=True):
        prior_shape = self.netShape.getMesh(total_iter=total_iter, jitter_grid=is_training)
        return prior_shape, self.netDINO
