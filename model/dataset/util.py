import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
from einops import rearrange


def compute_distance_transform(mask):
    mask_dt = []
    for m in mask:
        dt = torch.FloatTensor(cv2.distanceTransform(np.uint8(m[0]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        inv_dt = torch.FloatTensor(cv2.distanceTransform(np.uint8(1 - m[0]), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        mask_dt += [torch.stack([dt, inv_dt], 0)]
    return torch.stack(mask_dt, 0)  # Bx2xHxW


def crop_image(image, boxs, size):
    crops = []
    for box in boxs:
        crop_x0, crop_y0, crop_w, crop_h = box
        crop = transforms.functional.resized_crop(image, crop_y0, crop_x0, crop_h, crop_w, size)
        crop = transforms.functional.to_tensor(crop)
        crops += [crop]
    return torch.stack(crops, 0)  # BxCxHxW


def box_loader(fpath):
    box = np.loadtxt(fpath, 'str')
    box[0] = box[0].split('_')[0]
    box = box[:8]
    return box.astype(np.float32)


def read_feat_from_img(path, n_channels):
    feat = np.array(Image.open(path))
    return dencode_feat_from_img(feat, n_channels)


def dencode_feat_from_img(img, n_channels):
    n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
    n_tiles = int((n_channels + n_addon_channels) / 3)
    feat = rearrange(img, 'h (t w) c -> h w (t c)', t=n_tiles, c=3)
    feat = feat[:, :, :-n_addon_channels]
    feat = feat.astype('float32') / 255
    return feat.transpose(2, 0, 1)  # CxHxW


def dino_loader(fpath, n_channels):
    dino_map = read_feat_from_img(fpath, n_channels)
    return dino_map


def get_valid_mask(boxs, image_size):
    valid_masks = []
    for box in boxs:
        crop_x0, crop_y0, crop_w, crop_h, full_w, full_h = box[1:7].int().numpy()
        margin_w = int(crop_w * 0.02)  # discard a small margin near the boundary
        margin_h = int(crop_h * 0.02)
        mask_full = torch.ones(full_h-margin_h*2, full_w-margin_w*2)
        mask_full_pad = torch.nn.functional.pad(mask_full, (crop_w+margin_w, crop_w+margin_w, crop_h+margin_h, crop_h+margin_h), mode='constant', value=0.0)
        mask_full_crop = mask_full_pad[crop_y0+crop_h:crop_y0+crop_h*2, crop_x0+crop_w:crop_x0+crop_w*2]
        mask_crop = torch.nn.functional.interpolate(mask_full_crop[None, None, :, :], image_size, mode='nearest')[0,0]
        valid_masks += [mask_crop]
    return torch.stack(valid_masks, 0)  # NxHxW


def horizontal_flip_box(box):
    frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = box.unbind(1)
    box[:,1] = full_w - crop_x0 - crop_w  # x0
    return box


def none_to_nan(x):
    return torch.FloatTensor([float('nan')]) if x is None else x
