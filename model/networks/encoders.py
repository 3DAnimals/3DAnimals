__all__ = ['Encoder', 'Encoder32', 'VGGEncoder', 'ResnetEncoder', 'ViTEncoder', 'ResnetDepthEncoder']

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from .util import get_activation


class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=128, zdim=None, nf=64, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 128x128 -> 64x64
            nn.GroupNorm(16, nf),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16*2, nf*2),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*4, nf*4),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            # nn.GroupNorm(16*8, nf*8),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        add_downsample = int(np.log2(in_size//128))
        if add_downsample > 0:
            for _ in range(add_downsample):
                network += [
                    nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
                    # nn.GroupNorm(16*8, nf*8),
                    # nn.ReLU(inplace=True),
                    nn.LeakyReLU(0.2, inplace=True),
                ]

        network += [
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
        ]

        if zdim is None:
            network += [
                nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                ]
        else:
            network += [
                nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
                ]

        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class Encoder32(nn.Module):
    def __init__(self, cin, cout, nf=256, activation=None):
        super().__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.GroupNorm(nf//4, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, cout, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
        ]
        if activation is not None:
            network += [get_activation(activation)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class VGGEncoder(nn.Module):
    def __init__(self, cout, pretrained=False):
        super().__init__()
        if pretrained:
            raise NotImplementedError
        vgg = models.vgg16()
        self.vgg_encoder = nn.Sequential(vgg.features, vgg.avgpool)
        self.linear1 = nn.Linear(25088, 4096)
        self.linear2 = nn.Linear(4096, cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        out = self.relu(self.linear1(self.vgg_encoder(x).view(batch_size, -1)))
        return self.linear2(out)


class ResnetEncoder(nn.Module):
    def __init__(self, cout, pretrained=False):
        super().__init__()
        self.resnet = nn.Sequential(list(models.resnet18(weights="DEFAULT" if pretrained else None).modules())[:-1])
        self.final_linear = nn.Linear(512, cout)

    def forward(self, x):
        return self.final_linear(self.resnet(x))


class ResnetDepthEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.model = nn.Sequential(*list(self.resnet.children())[:-1])
        self.layer_idx_dict = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.float()),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.features = None
        if pretrained:
            for p in self.model.parameters():
                p.requires_grad = False

    def hook_fn(self, module, input, output):
        self.features = output

    def forward(self, x):
        # Input is 3 channel depth image
        self.features = None
        hook_handle = self.model[self.layer_idx_dict["layer2"]].register_forward_hook(self.hook_fn)
        global_feat = self.model(self.transform(x)).squeeze(-1).squeeze(-1)
        local_feat = self.features
        hook_handle.remove()
        return global_feat, local_feat


class ViTEncoder(nn.Module):
    def __init__(self, cout, which_vit='dino_vits8', pretrained=False, frozen=False, final_layer_type='none'):
        super().__init__()
        self.ViT = torch.hub.load('facebookresearch/dino:main', which_vit, pretrained=pretrained)
        if frozen:
            for p in self.ViT.parameters():
                p.requires_grad = False
        if which_vit == 'dino_vits8':
            self.vit_feat_dim = 384
            self.patch_size = 8
        elif which_vit == 'dino_vitb8':
            self.vit_feat_dim = 768
            self.patch_size = 8
        else:
            raise NotImplementedError
        
        self._feats = []
        self.hook_handlers = []

        if final_layer_type == 'none':
            pass
        elif final_layer_type == 'conv':
            self.final_layer_patch_out = Encoder32(self.vit_feat_dim, cout, nf=256, activation=None)
            self.final_layer_patch_key = Encoder32(self.vit_feat_dim, cout, nf=256, activation=None)
        elif final_layer_type == 'attention':
            raise NotImplementedError
            self.final_layer = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.fc = nn.Linear(self.vit_feat_dim, cout)
        else:
            raise NotImplementedError
        self.final_layer_type = final_layer_type
    
    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook
    
    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.ViT.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")
    
    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []
    
    def forward(self, x, return_patches=False):
        b, c, h, w = x.shape
        self._feats = []
        self._register_hooks([11], 'key')

        x = self.ViT.prepare_tokens(x)
        for blk in self.ViT.blocks:
            x = blk(x)
        out = self.ViT.norm(x)
        self._unregister_hooks()

        ph, pw = h // self.patch_size, w // self.patch_size
        patch_out = out[:, 1:]  # first is class token
        patch_out = patch_out.reshape(b, ph, pw, self.vit_feat_dim).permute(0, 3, 1, 2)

        patch_key = self._feats[0][:,:,1:]  # B, num_heads, num_patches, dim
        patch_key = patch_key.permute(0, 1, 3, 2).reshape(b, self.vit_feat_dim, ph, pw)

        if self.final_layer_type == 'none':
            global_feat_out = out[:, 0].reshape(b, -1)  # first is class token
            global_feat_key = self._feats[0][:, :, 0].reshape(b, -1)  # first is class token
        elif self.final_layer_type == 'conv':
            global_feat_out = self.final_layer_patch_out(patch_out).view(b, -1)
            global_feat_key = self.final_layer_patch_key(patch_key).view(b, -1)
        elif self.final_layer_type == 'attention':
            raise NotImplementedError
        else:
            raise NotImplementedError
        if not return_patches:
            patch_out = patch_key = None
        return global_feat_out, global_feat_key, patch_out, patch_key
