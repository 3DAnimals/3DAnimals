__all__ = ['ArticulationNetwork']

import torch
import torch.nn as nn
from .util import get_activation
from .HarmonicEmbedding import HarmonicEmbedding
from .MLPs import MLP


class ArticulationNetwork(nn.Module):
    def __init__(self, net_type, feat_dim, posenc_dim, num_layers, nf, n_harmonic_functions=0, embedder_scalar=1, activation=None, enable_articulation_idadd=False):
        super().__init__()
        if n_harmonic_functions > 0:
            self.posenc = HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions, scalar=embedder_scalar)
            posenc_dim = posenc_dim * (n_harmonic_functions * 2 + 1)
        else:
            self.posenc = None
            posenc_dim = 4
        cout = 3
        
        if net_type == 'mlp':
            self.network = MLP(
                feat_dim + posenc_dim,  # + bone xyz pos and index
                cout,  # We represent the rotation of each bone by its Euler angles ψ, θ, and φ
                num_layers,
                nf=nf,
                dropout=0,
                activation=activation
            )
        elif net_type == 'attention':
            self.in_layer = nn.Sequential(
                nn.Linear(feat_dim + posenc_dim, nf),
                nn.GELU(),
                nn.LayerNorm(nf),
            )
            self.blocks = nn.ModuleList([
            Block(
                dim=nf, num_heads=8, mlp_ratio=2., qkv_bias=False, qk_scale=None,
                drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm)
            for i in range(num_layers)])
            out_layer = [nn.Linear(nf, cout)]
            if activation:
                out_layer += [get_activation(activation)]
            self.out_layer = nn.Sequential(*out_layer)
        else:
            raise NotImplementedError
        self.net_type = net_type
        self.enable_articulation_idadd = enable_articulation_idadd
    
    def forward(self, x, pos):
        pos_inp = pos
        if self.posenc is not None:
            pos = torch.cat([pos, self.posenc(pos)], dim=-1)
        x = torch.cat([x, pos], dim=-1)
        if self.enable_articulation_idadd:
            articulation_id = pos_inp[..., -1:]
            x = x + articulation_id
        if self.net_type == 'mlp':
            out = self.network(x)
        elif self.net_type == 'attention':
            x = self.in_layer(x)
            for blk in self.blocks:
                x = blk(x)
            out = self.out_layer(x)
        else:
            raise NotImplementedError
        return out


## Attention block from ViT (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
