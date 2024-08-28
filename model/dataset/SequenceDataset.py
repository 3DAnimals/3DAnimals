import os
from pathlib import Path
from glob import glob
import random
import re
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode
from .util import *


class BaseSequenceDataset(Dataset):
    def __init__(self, root, skip_beginning=4, skip_end=4, min_seq_len=10):
        super().__init__()

        self.skip_beginning = skip_beginning
        self.skip_end = skip_end
        self.min_seq_len = min_seq_len
        self.sequences = self._make_sequences(root)
        self.samples = []

    def _make_sequences(self, path):
        result = []
        for d in sorted(os.scandir(path), key=lambda e: e.name):
            if d.is_dir():
                files = self._parse_folder(d)
                if len(files) >= self.min_seq_len:
                    result.append(files)
        return result

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '*'+image_path_suffix)))
        img_ext = [os.path.splitext(image_path_suffix)[-1]]
        if ".*" in img_ext:
            img_ext = set([os.path.splitext(r)[-1] for r in result])
        all_result = []
        for ext in img_ext:
            all_result += [p.replace(image_path_suffix.replace(".*", ext), '{}') for p in result if ext in p]
        all_result = sorted(list(set(all_result)))
        if len(all_result) <= self.skip_beginning + self.skip_end:
            return []
        if self.skip_end == 0:
            return all_result[self.skip_beginning:]
        return all_result[self.skip_beginning:-self.skip_end]

    def _load_ids(self, path_patterns, loader, transform=None):
        result = []
        for p in path_patterns:
            suffix = loader[0]
            if ".*" in suffix:
                re_pattern = re.compile(p.format(suffix))
                all_occurrence = [str(f) for f in Path(p).parent.iterdir() if re.search(re_pattern, str(f))]
                suffix = suffix.replace(".*", os.path.splitext(all_occurrence[0])[-1])
            x = loader[1](p.format(suffix), *loader[2:])
            if transform:
                x = transform(x)
            result.append(x)
        return tuple(result)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raise NotImplemented("This is a base class and should not be used directly")


class NFrameSequenceDataset(BaseSequenceDataset):
    def __init__(self, root, num_frames=2, skip_beginning=4, skip_end=4, min_seq_len=10, in_image_size=256, out_image_size=256, random_sample=False, dense_sample=True, shuffle=False, load_flow=False, load_background=False, random_xflip=False, load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64):
        self.image_loader = ["rgb.*", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.bbox_loader = ["box.txt", box_loader]
        super().__init__(root, skip_beginning, skip_end, min_seq_len)
        if load_flow and num_frames > 1:
            self.flow_loader = ["flow.png", cv2.imread, cv2.IMREAD_UNCHANGED]
        else:
            self.flow_loader = None

        self.num_frames = num_frames
        self.random_sample = random_sample
        if self.random_sample:
            self.samples = self.sequences
        else:
            for i, s in enumerate(self.sequences):
                stride = 1 if dense_sample else self.num_frames
                self.samples += [(i, k) for k in range(0, len(s), stride)]
        if shuffle:
            random.shuffle(self.samples)
        
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose([transforms.Resize(self.in_image_size, interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        if self.flow_loader is not None:
            def flow_transform(x):
                x = torch.FloatTensor(x.astype(np.float32)).flip(2)[:,:,:2]  # HxWx2
                x = x / 65535. * 2 - 1  # -1~1
                x = torch.nn.functional.interpolate(x.permute(2,0,1)[None], size=self.out_image_size, mode="bilinear")[0]  # 2xHxW
                return x
            self.flow_transform = flow_transform
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feat{dino_feature_dim}.png", dino_loader, dino_feature_dim]
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        self.load_flow = load_flow
        self.load_background = load_background
        self.random_xflip = random_xflip

    def __getitem__(self, index):
        if self.random_sample:
            seq_idx = index % len(self.samples)
            seq = self.samples[seq_idx]
            if len(seq) < self.num_frames:
                start_frame_idx = 0
            else:
                start_frame_idx = np.random.randint(len(seq)-self.num_frames+1)
        else:
            seq_idx, start_frame_idx = self.samples[index % len(self.samples)]
            seq = self.sequences[seq_idx]
            ## handle edge case: when only last frame is left, sample last two frames, except if the sequence only has one frame
            if len(seq) <= start_frame_idx +1:
                start_frame_idx = max(0, start_frame_idx-1)
        
        paths = seq[start_frame_idx:start_frame_idx+self.num_frames]  # length can be shorter than num_frames
        images = torch.stack(self._load_ids(paths, self.image_loader, transform=self.image_transform), 0)  # load all images
        masks = torch.stack(self._load_ids(paths, self.mask_loader, transform=self.mask_transform), 0)  # load all images
        mask_dt = compute_distance_transform(masks)
        bboxs = torch.stack(self._load_ids(paths, self.bbox_loader, transform=torch.FloatTensor), 0)   # load bounding boxes for all images
        mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))  # exclude pixels cropped outside the original image
        if self.load_flow and len(paths) > 1:
            flows = torch.stack(self._load_ids(paths[:-1], self.flow_loader, transform=self.flow_transform), 0)  # load flow from current frame to next, (N-1)x(x,y)xHxW, -1~1
        else:
            flows = None
        if self.load_background:
            bg_fpath = os.path.join(os.path.dirname(paths[0]), 'background_frame.jpg')
            assert os.path.isfile(bg_fpath)
            bg_image = torchvision.datasets.folder.default_loader(bg_fpath)
            bg_images = crop_image(bg_image, bboxs[:, 1:5].int().numpy(), (self.out_image_size, self.out_image_size))
        else:
            bg_images = None
        if self.load_dino_feature:
            dino_features = torch.stack(self._load_ids(paths, self.dino_feature_loader, transform=torch.FloatTensor), 0)  # Fx64x224x224
        else:
            dino_features = None
        if self.load_dino_cluster:
            dino_clusters = torch.stack(self._load_ids(paths, self.dino_cluster_loader, transform=transforms.ToTensor()), 0)  # Fx3x55x55
        else:
            dino_clusters = None
        seq_idx = torch.LongTensor([seq_idx])
        frame_idx = torch.arange(start_frame_idx, start_frame_idx+len(paths)).long()

        ## random horizontal flip
        if self.random_xflip and np.random.rand() < 0.5:
            xflip = lambda x: None if x is None else x.flip(-1)
            images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters = (*map(xflip, (images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters)),)
            if flows is not None:
                flows[:,0] *= -1  # invert delta x
            bboxs = horizontal_flip_box(bboxs)  # NxK

        ## pad shorter sequence
        if len(paths) < self.num_frames:
            num_pad = self.num_frames - len(paths)
            pad_front = lambda x: None if x is None else torch.cat([x[:1]] *num_pad + [x], 0)
            images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, frame_idx = (*map(pad_front, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters,frame_idx)),)
            if flows is not None:
                flows[:num_pad] = 0  # setting flow to zeros for replicated frames

        out = (*map(none_to_nan, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, seq_idx, frame_idx)),)  # for batch collation
        return out
