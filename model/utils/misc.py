import os
import glob
import yaml
import random
import numpy as np
import cv2
import torch
import torchvision.utils as tvutils
import zipfile
from omegaconf.errors import ConfigAttributeError
from dataclasses import fields, is_dataclass
from ..render.obj import write_obj


def setup_runtime(cfg):
    """Load configs, initialize CUDA, CuDNN and the random seeds."""

    # Setup CUDA
    cuda_device_id = cfg.gpu
    if cuda_device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device_id)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    device = 'cuda:0' if torch.cuda.is_available() and cuda_device_id is not None else 'cpu'

    # Setup random seeds for reproducibility
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Environment: GPU {cuda_device_id} - seed {seed}")
    return device


def load_yaml(path):
    print(f"Loading configs from {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(path, cfgs):
    print(f"Saving configs to {path}")
    xmkdir(os.path.dirname(path))
    with open(path, 'w') as f:
        return yaml.safe_dump(cfgs, f)


def xmkdir(path):
    """Create directory PATH recursively if it does not exist."""
    os.makedirs(path, exist_ok=True)


def clean_checkpoint(checkpoint_dir, keep_num=2):
    if keep_num > 0:
        names = list(sorted(
            glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth')),
            key=lambda x: int(''.join([c for c in x if c.isdigit()]))
        ))
        if len(names) > keep_num:
            for name in names[:-keep_num]:
                print(f"Deleting obslete checkpoint file {name}")
                try:
                    os.remove(name)
                except FileNotFoundError:
                    pass


def archive_code(arc_path, filetypes=['.py']):
    print(f"Archiving code to {arc_path}")
    xmkdir(os.path.dirname(arc_path))
    zipf = zipfile.ZipFile(arc_path, 'w', zipfile.ZIP_DEFLATED)
    cur_dir = os.getcwd()
    flist = []
    for ftype in filetypes:
        flist.extend(glob.glob(os.path.join(cur_dir, '[!results]*', '**', '*'+ftype), recursive=True))  # ignore results folder
        flist.extend(glob.glob(os.path.join(cur_dir, '*'+ftype)))
    [zipf.write(f, arcname=f.replace(cur_dir,'archived_code', 1)) for f in flist]
    zipf.close()


def get_model_device(model):
    return next(model.parameters()).device


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_videos(out_fold, imgs, prefix='', suffix='', fnames=None, ext='.mp4', cycle=False):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    imgs = imgs.transpose(0, 1, 3, 4, 2)  # BxTxCxHxW -> BxTxHxWxC
    for i, fs in enumerate(imgs):
        if cycle:
            fs = np.concatenate([fs, fs[::-1]], 0)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # fourcc = cv2.VideoWriter_fourcc(*'avc1')

        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ext))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix + fname + suffix + ext)

        vid = cv2.VideoWriter(fpath, fourcc, 5, (fs.shape[2], fs.shape[1]))
        [vid.write(np.uint8(f[..., ::-1] * 255.)) for f in fs]
        vid.release()


def save_images(out_fold, imgs, prefix='', suffix='', fnames=None, ext='.png'):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    imgs = imgs.transpose(0, 2, 3, 1)
    for i, img in enumerate(imgs):
        img = np.concatenate([np.flip(img[..., :3], -1), img[..., 3:]], -1)  # RGBA to BGRA
        if 'depth' in suffix:
            im_out = np.uint16(img * 65535.)
        else:
            im_out = np.uint8(img * 255.)

        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ext))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix + fname + suffix + ext)

        cv2.imwrite(fpath, im_out)


def save_txt(out_fold, data, prefix='', suffix='', fnames=None, ext='.txt', fmt='%.6f', delim=', '):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    for i, d in enumerate(data):
        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ext))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        fpath = os.path.join(out_fold_i, prefix + fname + suffix + ext)

        np.savetxt(fpath, d, fmt=fmt, delimiter=delim)


def save_obj(out_fold, meshes=None, save_material=True, feat=None, prefix='', suffix='', fnames=None, resolution=[256, 256]):
    prefix = prefix + '_' if prefix else ''
    suffix = '_' + suffix if suffix else ''

    if meshes.v_pos is None:
        return

    batch_size = meshes.v_pos.shape[0]
    for i in range(batch_size):
        out_fold_i = out_fold[i] if isinstance(out_fold, list) else out_fold
        xmkdir(out_fold_i)

        if fnames is None:
            idx = len(glob.glob(os.path.join(out_fold_i, prefix + '*' + suffix + ".obj"))) + 1
            fname = '%07d' % idx
        else:
            fname = fnames[i]
        write_obj(out_fold_i, prefix+fname+suffix, meshes, i, save_material=save_material, feat=feat, resolution=resolution)


def compute_sc_inv_err(d_pred, d_gt, mask=None):
    b = d_pred.size(0)
    diff = d_pred - d_gt
    if mask is not None:
        diff = diff * mask
        avg = diff.view(b, -1).sum(1) / (mask.view(b, -1).sum(1))
        score = (diff - avg.view(b, 1, 1)) ** 2 * mask
    else:
        avg = diff.view(b, -1).mean(1)
        score = (diff - avg.view(b, 1, 1)) ** 2
    return score  # masked error maps


def compute_angular_distance(n1, n2, mask=None):
    dist = (n1 * n2).sum(3).clamp(-1, 1).acos() / np.pi * 180
    return dist * mask if mask is not None else dist


def save_scores(out_path, scores, header=''):
    print('Saving scores to %s' % out_path)
    np.savetxt(out_path, scores, fmt='%.8f', delimiter=',\t', header=header)


def image_grid(tensor, nrow=None):
    b, c, h, w = tensor.shape
    if nrow is None:
        nrow = int(np.ceil(b ** 0.5))
    if c == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    tensor = tvutils.make_grid(tensor, nrow=nrow, normalize=False)
    return tensor


def video_grid(tensor, nrow=None):
    return torch.stack([image_grid(t, nrow=nrow) for t in tensor.unbind(1)], 0)


def in_range(x, range, default_indicator=None):
    range_min, range_max = range
    if isinstance(range_max, str):
        range_max = float(range_max)
    if isinstance(range_min, str):
        range_min = float(range_min)
    min_check = (x >= range_min)
    max_check = (x < range_max)
    if default_indicator is not None:
        if range_min == default_indicator:
            min_check = True
        if range_max == default_indicator:
            max_check = True
    return min_check and max_check


def load_cfg(self, cfg, config_class):
    """Load configs defined in config_class only and set attributes in self, recurse if a field is a dataclass"""
    cfg_dict = {}
    for field in fields(config_class):
        if is_dataclass(field.type):  # Recurse if field is dataclass
            value = load_cfg(None, getattr(cfg, field.name), field.type)
        else:
            try:
                value = getattr(cfg, field.name)
            except ConfigAttributeError:
                print(f"{config_class.__name__}.{field.name} not in config, using default value: {field.default}")
                continue
        cfg_dict[field.name] = value
    cfg = config_class(**cfg_dict)
    if self is not None:
        self.cfg = cfg
        for field in fields(cfg):
            setattr(self, field.name, getattr(cfg, field.name))
    return cfg


def add_text_to_image(img, text, pos=(12, 12), color=(1, 1, 1), font_scale=0.5, thickness=1):
    if isinstance(img, torch.Tensor):
        img = img.permute(1,2,0).cpu().numpy()
    # if grayscale -> convert to RGB
    if img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
    img = cv2.putText(np.ascontiguousarray(img), text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def normalize_depth(depth, mask=None):
    # Normalize depth values to [0, 1] range based on the minimum and maximum depth values in masked regions
    # Set all background to 0
    if mask is not None:
        depth_for_min = torch.where(mask.bool(), depth, torch.full_like(depth, float('inf')))
        depth_for_max = torch.where(mask.bool(), depth, torch.full_like(depth, -float('inf')))
    else:
        depth_for_min = depth
        depth_for_max = depth
    depth_min = depth_for_min.amin(dim=(-1, -2), keepdim=True)
    depth_max = depth_for_max.amax(dim=(-1, -2), keepdim=True)
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    if mask is not None:
        normalized_depth = torch.where(mask.bool(), normalized_depth, torch.zeros_like(normalized_depth))
    return normalized_depth
