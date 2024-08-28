import os
from glob import glob
import random
import re
import copy

import torch
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode
from .util import *

'''
Combine categories/paths in:
- data_dir/{few_shot_animal3d, few_shot_web, few_shot_web_back, large_scale}
'''

LARGE_SCALE_NAMES = [
    'large_scale'
]

SMALL_SCALE_NAMES = [
    'few_shot_animal3d',
    'few_shot_web',
    'few_shot_web_back'
]


def small_scale_box_loader(fpath):
    box = np.loadtxt(fpath, 'str')
    # box[0] = box[0].split('_')[0]
    return box.astype(np.float32)


def horizontal_flip_box_new(box):
    frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, tmp_label = box.unbind(1)
    box[:,1] = full_w - crop_x0 - crop_w  # x0
    return box


class FaunaDataset(Dataset):
    def __init__(self, root, in_image_size=256, out_image_size=256, shuffle=False, load_background=False, random_xflip=False, 
                 load_dino_feature=False, load_dino_cluster=False, dino_feature_dim=64, split='train', batch_size=6,
                 dataset_split_num=-1):
        super().__init__()

        self.split = split
        self.val_num = 5

        large_scale_paths = dict()
        for large_scale_name in LARGE_SCALE_NAMES:
            category_names = sorted(os.listdir(os.path.join(root, large_scale_name)))
            for category_name in category_names:
                large_scale_path = os.path.join(root, large_scale_name, category_name, self.split)
                save_name = category_name.split('_')[0]
                large_scale_paths.update({
                    f'large_scale_{save_name}': large_scale_path
                })
        self.large_scale_paths = large_scale_paths
        
        small_scale_paths = dict()
        small_scale_with_back = []
        for small_scale_name in SMALL_SCALE_NAMES:
            if small_scale_name.endswith("_back"):
                small_scale_with_back.append(small_scale_name[:-5].split('_')[-1])
                continue
            
            category_names = sorted(os.listdir(os.path.join(root, small_scale_name)))
            for category_name in category_names:
                small_scale_path = os.path.join(root, small_scale_name, category_name, 'train')     # for small scale, we split the images for train/val&test
                save_name = category_name
                save_name_prefix = small_scale_name.split('_')[-1]
                small_scale_paths.update({
                    f'small_scale_{save_name_prefix}_{save_name}': small_scale_path
                })
        self.small_scale_paths = small_scale_paths
        self.small_scale_with_back = small_scale_with_back
        
        self.image_loader = ["rgb.*", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.large_scale_bbox_loader = ["box.txt", box_loader]
        self.small_scale_bbox_loader = ["box.txt", small_scale_box_loader]
        # in fauna dataset we use batch_size to pad each category to make each batch contains same categories
        self.batch_size = batch_size

        large_scale_data_paths = {}
        for k,v in self.large_scale_paths.items():
            sequences = self._make_sequences(v)
            samples = []
            for seq in sequences:
                samples += seq
            # if shuffle:
            if self.split == 'train':
                random.shuffle(samples)
            large_scale_data_paths.update({k: samples})
        
        small_scale_data_paths = {}
        for k,v in self.small_scale_paths.items():
            result = sorted(glob(os.path.join(v, '*'+self.image_loader[0])))
            result = [p.replace(self.image_loader[0], '{}') for p in result]
            sequences = result
            
            if k.split('_')[2] in self.small_scale_with_back:
                back_dataset_name = v.split('/')[-3] + '_back'
                back_category_name = v.split('/')[-2]
                back_view_dir = os.path.join(root, back_dataset_name, back_category_name, 'train')
                back_view_result = sorted(glob(os.path.join(back_view_dir, '*'+self.image_loader[0])))
                back_view_result = [p.replace(self.image_loader[0], '{}') for p in back_view_result]
                mul_bv_sequences = self._more_back_views(back_view_result, result)
                sequences = mul_bv_sequences + sequences
            
            if split != 'train':
                sequences = sequences[-self.val_num:]
            else:
                sequences = sequences[:-self.val_num]
            
            small_scale_data_paths.update({k: sequences})
        
        self.small_scale_data_length = self._get_data_length(small_scale_data_paths)

        print_categories = list(large_scale_data_paths.keys()) + list(small_scale_data_paths)
        print_caategory_nums = len(large_scale_data_paths.keys()) + len(small_scale_data_paths.keys())
        print(f"using {print_caategory_nums} categories, contains: {print_categories}")

        self.dataset_split_num = dataset_split_num # if -1 then pad to longest, otherwise follow this number to pad and split
        if split != 'train':
            self.dataset_split_num = -1            # validation we don't split dataset
        
        if self.dataset_split_num == -1:
            self.all_data_paths, self.one_category_num = self._pad_paths(large_scale_data_paths, small_scale_data_paths)
            self.all_category_num = len(self.all_data_paths.keys())
            self.all_category_names = list(self.all_data_paths.keys())
            self.original_category_names = list(self.large_scale_paths.keys())
        elif self.dataset_split_num > 0:
            self.all_data_paths, self.one_category_num, self.original_category_names = self._pad_paths_withnum(large_scale_data_paths, small_scale_data_paths, self.dataset_split_num)
            self.all_category_num = len(self.all_data_paths.keys())
            self.all_category_names = list(self.all_data_paths.keys())
        else:
            raise NotImplementedError
        
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
    
    def __len__(self):
        return self.all_category_num * self.one_category_num
    
    def __getitem__(self, index):
        category_idx = (index % (self.batch_size * self.all_category_num)) // self.batch_size
        path_idx = (index // (self.batch_size * self.all_category_num)) * self.batch_size + (index % (self.batch_size * self.all_category_num)) - category_idx * self.batch_size
        category_name = self.all_category_names[category_idx]
        path = self.all_data_paths[category_name][path_idx]
        images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)

        if category_name.startswith("large_scale_"):
            bbox_loader = self.large_scale_bbox_loader
            use_original_bbox = True
        else:
            bbox_loader = self.small_scale_bbox_loader
            use_original_bbox = False

        masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
        mask_dt = compute_distance_transform(masks)
        bboxs = self._load_ids(path, bbox_loader, transform=torch.FloatTensor).unsqueeze(0)
        bboxs = torch.cat([bboxs, torch.Tensor([[category_idx]]).float()], dim=-1)  # pad a label number

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
            bboxs = horizontal_flip_box_new(bboxs)  # NxK
        out = (*map(none_to_nan, (images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, seq_idx, frame_idx)),)  # for batch collation
        return out

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def _shuffle_all(self):
        for k,v in self.all_data_paths.items():
            new_v = copy.deepcopy(v)
            random.shuffle(new_v)
            self.all_data_paths[k] = new_v
        return None
        
    def _pad_paths(self, ori_paths, fs_paths):
        img_nums = []
        all_paths = copy.deepcopy(ori_paths)
        all_paths.update(fs_paths)
        for _, v in all_paths.items():
            img_nums.append(len(v))
        
        img_num = max(img_nums)
        img_num = (img_num // self.batch_size) * self.batch_size

        for k,v in all_paths.items():
            if len(v) < img_num:
                mul_time = img_num // len(v)
                pad_time = img_num % len(v)
                # for each v, shuffle it
                shuffle_v = copy.deepcopy(v)
                new_v = []
                for i in range(mul_time):
                    new_v = new_v + shuffle_v
                    random.shuffle(shuffle_v)
                del shuffle_v
                new_v = new_v + v[0:pad_time]
                # new_v = mul_time * v + v[0:pad_time]
                all_paths[k] = new_v
            elif len(v) > img_num:
                all_paths[k] = v[:img_num]
            else:
                continue
        
        return all_paths, img_num

    def _pad_paths_withnum(self, ori_paths, fs_paths, split_num=1000):
        img_num = (split_num // self.batch_size) * self.batch_size
        all_paths = {}
        orig_cat_names = []

        for k, v in ori_paths.items():
            total_num = ((len(v) // img_num) + 1) * img_num
            pad_num = total_num - len(v)
            split_num = total_num // img_num

            new_v = copy.deepcopy(v)
            random.shuffle(new_v)
            all_v = v + new_v[:pad_num]
            del new_v

            for sn in range(split_num):
                split_cat_name = f'{k}_' + '%03d' % sn
                all_paths.update({
                    split_cat_name: all_v[sn*img_num: (sn+1)*img_num]
                })
                orig_cat_names.append(split_cat_name)
        
        for k, v in fs_paths.items():
            if len(v) < img_num:
                mul_time = img_num // len(v)
                pad_time = img_num % len(v)
                # for each v, shuffle it
                shuffle_v = copy.deepcopy(v)
                new_v = []
                for i in range(mul_time):
                    new_v = new_v + shuffle_v
                    random.shuffle(shuffle_v)
                del shuffle_v
                new_v = new_v + v[0:pad_time]
                # new_v = mul_time * v + v[0:pad_time]
                all_paths.update({
                    k: new_v
                })
            elif len(v) > img_num:
                all_paths.update({
                    k: v[:img_num]
                })
            else:
                continue
        
        return all_paths, img_num, orig_cat_names
    
    def _make_sequences(self, path):
        result = []
        for d in sorted(os.scandir(path), key=lambda e: e.name):
            if d.is_dir():
                files = self._parse_folder(d)
                if len(files) >= 1:
                    result.append(files)
        return result

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = sorted(glob(os.path.join(path, '*'+image_path_suffix)))
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]
        return result
    
    def _more_back_views(self, back_view_seq, seq):
        if len(back_view_seq) == 0:
            # for category without back views
            return []
        factor = 5
        # length = (len(seq) // factor) * factor
        length = (len(seq) // factor) * (factor - 1)
        mul_f = length // len(back_view_seq)
        pad_f = length % len(back_view_seq)
        new_seq = mul_f * back_view_seq + back_view_seq[:pad_f]
        return new_seq
    
    def _get_data_length(self, paths):
        data_length = {}
        for k,v in paths.items():
            length = len(v)
            data_length.update({k: length})
        return data_length
