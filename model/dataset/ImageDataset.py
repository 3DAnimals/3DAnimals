import os
from glob import glob
import random
import re
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode
from .util import *


class ImageDataset(Dataset):
    def __init__(self, root, in_image_size=256, out_image_size=256, shuffle=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
        super().__init__()
        self.image_loader = ["rgb.*", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.bbox_loader = ["box.txt", box_loader]
        self.samples = self._parse_folder(root)
        if shuffle:
            random.shuffle(self.samples)
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose([transforms.Resize(self.in_image_size, interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feat{dino_feature_dim}.png", dino_loader, dino_feature_dim]
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        self.load_background = load_background
        self.random_xflip = random_xflip

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '**/*'+image_path_suffix), recursive=True))
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]
        return result

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def __len__(self):
        return len(self.samples)
    
    def set_random_xflip(self, random_xflip):
        self.random_xflip = random_xflip

    def __getitem__(self, index):
        path = self.samples[index % len(self.samples)]
        images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)
        masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
        mask_dt = compute_distance_transform(masks)
        bboxs = self._load_ids(path, self.bbox_loader, transform=torch.FloatTensor).unsqueeze(0)
        mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        flows = None
        if self.load_background:
            bg_fpath = os.path.join(os.path.dirname(path), 'background_frame.jpg')
            assert os.path.isfile(bg_fpath)
            bg_image = torchvision.datasets.folder.default_loader(bg_fpath)
            bg_images = crop_image(bg_image, bboxs[:, 1:5].int().numpy(), (self.out_image_size, self.out_image_size))
        else:
            bg_images = None
        if self.load_dino_feature:
            dino_features = self._load_ids(path, self.dino_feature_loader, transform=torch.FloatTensor).unsqueeze(0)
        else:
            dino_features = None
        if self.load_dino_cluster:
            dino_clusters = self._load_ids(path, self.dino_cluster_loader, transform=transforms.ToTensor()).unsqueeze(0)
        else:
            dino_clusters = None
        seq_idx = torch.LongTensor([index])
        frame_idx = torch.LongTensor([0])

        ## random horizontal flip
        if self.random_xflip and np.random.rand() < 0.5:
            xflip = lambda x: None if x is None else x.flip(-1)
            images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters = (*map(xflip, (images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters)),)
            bboxs = horizontal_flip_box(bboxs)  # NxK

        out = (*map(none_to_nan, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, seq_idx, frame_idx)),)  # for batch collation
        return out
