import os.path as osp
from dataclasses import dataclass
from .dataset import *


@dataclass
class DataLoaderConfig:
    data_type: str = 'image'
    in_image_size: int = 256
    out_image_size: int = 256
    batch_size: int = 64
    num_workers: int = 4
    train_data_dir: str = None
    val_data_dir: str = None
    test_data_dir: str = None
    random_shuffle_samples_train: bool = False
    random_xflip_train: bool = False
    load_flow: bool = False
    background_mode: str = 'none'  # none (black), white, checkerboard, background, input
    load_dino_feature: bool = False
    load_dino_cluster: bool = False
    dino_feature_dim: int = 64

    # sequence specific
    skip_beginning: int = 0
    skip_end: int = 0
    num_frames: int = 2
    min_seq_len: int = 2
    random_sample_frames_train: bool = False
    random_sample_frames_val: bool = False


def get_data_loaders(cfg: DataLoaderConfig, dataset_split_num=-1):
    train_loader = val_loader = test_loader = None
    load_background = cfg.background_mode == 'background'
    random_shuffle_samples = {
        'train': cfg.random_shuffle_samples_train,
        'val': False,
        'test': False
    }
    random_xflip = {
        'train': cfg.random_xflip_train,
        'val': False,
        'test': False
    }

    def get_loader(mode, data_dir):
        if cfg.data_type == 'sequence':
            random_sample_frames = {
                'train': cfg.random_sample_frames_train,
                'val': cfg.random_sample_frames_val,
                'test': False
            }
            dense_sample = {
                'train': True,
                'val': True,
                'test': False
            }
            dataset = NFrameSequenceDataset(
                data_dir,
                num_frames=cfg.num_frames,
                skip_beginning=cfg.skip_beginning,
                skip_end=cfg.skip_end,
                min_seq_len=cfg.min_seq_len,
                in_image_size=cfg.in_image_size,
                out_image_size=cfg.out_image_size,
                random_sample=random_sample_frames[mode],
                dense_sample=dense_sample[mode],
                shuffle=random_shuffle_samples[mode],
                load_flow=cfg.load_flow,
                load_background=load_background,
                random_xflip=random_xflip[mode],
                load_dino_feature=cfg.load_dino_feature,
                load_dino_cluster=cfg.load_dino_cluster,
                dino_feature_dim=cfg.dino_feature_dim)
        
        elif cfg.data_type == 'image':
            dataset = ImageDataset(
                data_dir,
                in_image_size=cfg.in_image_size,
                out_image_size=cfg.out_image_size,
                load_background=load_background,
                random_xflip=random_xflip[mode],
                load_dino_feature=cfg.load_dino_feature,
                load_dino_cluster=cfg.load_dino_cluster,
                dino_feature_dim=cfg.dino_feature_dim)
        
        elif cfg.data_type == 'fauna':
            dataset = FaunaDataset(
                data_dir,
                in_image_size=cfg.in_image_size,
                out_image_size=cfg.out_image_size,
                load_background=load_background,
                random_xflip=random_xflip[mode],
                load_dino_feature=cfg.load_dino_feature,
                load_dino_cluster=cfg.load_dino_cluster,
                dino_feature_dim=cfg.dino_feature_dim,
                split=mode,
                batch_size=cfg.batch_size,
                dataset_split_num=dataset_split_num
            )

        else:
            raise ValueError(f"Unexpected data type: {cfg.data_type}")
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=random_shuffle_samples[mode],
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        return loader

    if cfg.train_data_dir is not None:
        assert osp.isdir(cfg.train_data_dir), f"Training data directory does not exist: {cfg.train_data_dir}"
        print(f"Loading training data from {cfg.train_data_dir}")
        train_loader = get_loader(mode='train', data_dir=cfg.train_data_dir)

    if cfg.val_data_dir is not None:
        assert osp.isdir(cfg.val_data_dir), f"Validation data directory does not exist: {cfg.val_data_dir}"
        print(f"Loading validation data from {cfg.val_data_dir}")
        val_loader = get_loader(mode='val', data_dir=cfg.val_data_dir)

    if cfg.test_data_dir is not None:
        assert osp.isdir(cfg.test_data_dir), f"Testing data directory does not exist: {cfg.test_data_dir}"
        print(f"Loading testing data from {cfg.test_data_dir}")
        test_loader = get_loader(mode='test', data_dir=cfg.test_data_dir)

    return train_loader, val_loader, test_loader