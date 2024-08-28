__all__ = ['MLP', 'CoordMLP']

import torch
import torch.nn as nn
from .util import get_activation
from .HarmonicEmbedding import HarmonicEmbedding


class MLP(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        super().__init__()
        assert num_layers >= 1
        if num_layers == 1:
            network = [nn.Linear(cin, cout, bias=False)]
        else:
            network = [nn.Linear(cin, nf, bias=False)]
            for _ in range(num_layers-2):
                network += [
                    nn.ReLU(inplace=True),
                    nn.Linear(nf, nf, bias=False)]
                if dropout:
                    network += [nn.Dropout(dropout)]
            network += [
                nn.ReLU(inplace=True),
                nn.Linear(nf, cout, bias=False)]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class CoordMLP(nn.Module):
    def __init__(self,
                 cin,
                 cout,
                 num_layers,
                 nf=256,
                 dropout=0,
                 activation=None,
                 min_max=None,
                 n_harmonic_functions=10,
                 embedder_scalar=1,
                 embed_concat_pts=True,
                 extra_feat_dim=0,
                 symmetrize=False,
                 in_layer_relu=False):
        super().__init__()
        self.extra_feat_dim = extra_feat_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, embedder_scalar)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP(nf + extra_feat_dim, cout, num_layers, nf, dropout, activation)
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)  # Cx2
        else:
            self.min_max = None
        self.bsdf = None
        self.in_layer_relu = in_layer_relu

    def forward(self, x, feat=None):
        # x: (B, ..., 3), feat: (B, C)
        assert (feat is None and self.extra_feat_dim == 0) or (feat.shape[-1] == self.extra_feat_dim)
        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.in_layer(x_in)
        if self.in_layer_relu:
            x_in = self.relu(x_in)
        if feat is not None:
            for i in range(x_in.dim() - feat.dim()):
                feat = feat.unsqueeze(1)
            feat = feat.expand(*x_in.shape[:-1], -1)
            x_in = torch.concat([x_in, feat], dim=-1)
        out = self.mlp(self.relu(x_in))  # (B, ..., C)
        if self.min_max is not None:
            out = out * (self.min_max[:,1] - self.min_max[:,0]) + self.min_max[:,0]
        return out

    def sample(self, x, feat=None):
        return self.forward(x, feat)


class CoordMLP_Mod(nn.Module):
    def __init__(self,
                 cin,
                 cout,
                 num_layers,
                 nf=256,
                 dropout=0,
                 activation=None,
                 min_max=None,
                 n_harmonic_functions=10,
                 embedder_scalar=1,
                 embed_concat_pts=True,
                 extra_feat_dim=0,
                 symmetrize=False,
                 condition_dim=128):
        super().__init__()
        self.extra_feat_dim = extra_feat_dim
        self.condition_dim = condition_dim

        if n_harmonic_functions > 0:
            self.embedder = HarmonicEmbedding(n_harmonic_functions, embedder_scalar)
            dim_in = cin * 2 * n_harmonic_functions
            self.embed_concat_pts = embed_concat_pts
            if embed_concat_pts:
                dim_in += cin
        else:
            self.embedder = None
            dim_in = cin
        
        self.in_layer = nn.Linear(dim_in, nf)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = MLP_Mod(nf, cout, num_layers, nf, dropout, activation)
        self.style_mlp = MLP(condition_dim, nf, 2, nf, dropout, None)
        self.symmetrize = symmetrize
        if min_max is not None:
            self.register_buffer('min_max', min_max)  # Cx2
        else:
            self.min_max = None
        self.bsdf = None

    def forward(self, x, feat=None):
        # x: (B, ..., 3), feat: (B, C)
        # assert (feat is None and self.extra_feat_dim == 0) or (feat.shape[-1] == self.extra_feat_dim) or (feat.shape[-1] == self.condition_dim)
        assert feat is not None and (feat.shape[-1] == self.condition_dim)
        if self.symmetrize:
            xs, ys, zs = x.unbind(-1)
            x = torch.stack([xs.abs(), ys, zs], -1)  # mirror -x to +x
        
        if self.embedder is not None:
            x_in = self.embedder(x)
            if self.embed_concat_pts:
                x_in = torch.cat([x, x_in], -1)
        else:
            x_in = x
        
        x_in = self.relu(self.in_layer(x_in))
        
        # [B, C], in mlp_mod we use only one feature for the batch
        style = self.style_mlp(feat)
        out = self.mlp(x_in, style)  # (B, ..., C)
        if self.min_max is not None:
            out = out * (self.min_max[:,1] - self.min_max[:,0]) + self.min_max[:,0]
        return out

    def sample(self, x, feat=None):
        return self.forward(x, feat)


class MLP_Mod(nn.Module):
    def __init__(self, cin, cout, num_layers, nf=256, dropout=0, activation=None):
        # default no dropout
        super().__init__()
        assert num_layers >= 1
        self.num_layers = num_layers
        if num_layers == 1:
            self.network = Linear_Mod(cin, cout, bias=False)
        else:
            self.relu = nn.ReLU(inplace=True)
            for i in range(num_layers):
                if i == 0:
                    setattr(self, f'linear_{i}', Linear_Mod(cin, nf, bias=False))
                elif i == (num_layers-1):
                    setattr(self, f'linear_{i}', Linear_Mod(nf, cout, bias=False))
                else:
                    setattr(self, f'linear_{i}', Linear_Mod(nf, nf, bias=False))

    def forward(self, input, style):
        if self.num_layers == 1:
            out = self.network(input, style)
        else:
            x = input
            for i in range(self.num_layers):
                linear_layer = getattr(self, f'linear_{i}')
                if i == (self.num_layers - 1):
                    x = linear_layer(x, style)
                else:
                    x = linear_layer(x, style)
                    x = self.relu(x)
            
            out = x
        return out


import math


class Linear_Mod(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, style):
        # weight: [out_features, in_features]
        # style: [..., in_features]
        if len(style.shape) > 1:
            style = style.reshape(-1, style.shape[-1])
            style = style[0]
        
        weight = self.weight * style.unsqueeze(0)
        decoefs = ((weight * weight).sum(dim=-1, keepdim=True) + 1e-5).sqrt()
        weight = weight / decoefs

        return torch.nn.functional.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
