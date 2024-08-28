__all__ = ["HarmonicEmbedding"]

import torch
import torch.nn as nn


class HarmonicEmbedding(nn.Module):
    def __init__(self, n_harmonic_functions=10, scalar=1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]
        Note that `x` is also premultiplied by `scalar` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.frequencies = scalar * (2.0 ** torch.arange(n_harmonic_functions))

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
