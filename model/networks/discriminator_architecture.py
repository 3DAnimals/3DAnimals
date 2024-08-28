import torch.nn as nn
import torch
from math import log2
import torch.nn.functional as F
from torch import autograd


class DCDiscriminator(nn.Module):
    ''' DC Discriminator class.

    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, in_dim=1, out_dim=1, n_feat=512, img_size=256, last_bias=False):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        n_layers = int(log2(img_size) - 2)
        self.blocks = nn.ModuleList(
            [nn.Conv2d(
                in_dim,
                int(n_feat / (2 ** (n_layers - 1))),
                4, 2, 1, bias=False)] + [nn.Conv2d(
                    int(n_feat / (2 ** (n_layers - i))),
                    int(n_feat / (2 ** (n_layers - 1 - i))),
                    4, 2, 1, bias=False) for i in range(1, n_layers)])

        self.conv_out = nn.Conv2d(n_feat, out_dim, 4, 1, 0, bias=last_bias)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            import ipdb; ipdb.set_trace()
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))

        out = self.conv_out(x)
        out = out.reshape(batch_size, self.out_dim)
        return out


# class ADADiscriminator(DCDiscriminator):
#     def __init__(self, aug, aug_p, **kwargs):
#         super().__init__(**kwargs)
#         self.aug = build_from_config(aug)
#         self.aug.p.copy_(torch.tensor(aug_p, dtype=torch.float32))
#         self.resolution = kwargs['img_size']

#     def get_resolution(self):
#         return self.resolution

#     def forward(self, x, **kwargs):
#         x = self.aug(x)
#         return super().forward(x, **kwargs)


# class ADADiscriminatorView(ADADiscriminator):
#     def __init__(self, out_dim_position, out_dim_latent, **kwargs):
#         self.out_dim_position = out_dim_position
#         self.out_dim_latent = out_dim_latent

#         super().__init__(**kwargs)

def bce_loss_target(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss.mean()

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg.mean()