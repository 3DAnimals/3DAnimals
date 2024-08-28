# 3D Animals Codebase

## About
This is a repository that contains the unified framework of the animal reconstruction series, including：

- MagicPony: [project](https://3dmagicpony.github.io/), [paper](https://arxiv.org/abs/2211.12497)
- 3D-Fauna: [project](https://kyleleey.github.io/3DFauna/), [paper](https://arxiv.org/abs/2401.02400)
- Ponymation: [project](https://keqiangsun.github.io/projects/ponymation/), [paper](https://arxiv.org/abs/2312.13604)

## Setup (with [conda](https://docs.conda.io/en/latest/))

### 1. Install dependencies
```shell
conda env create -f environment.yml
```
or manually:
```shell
conda install -c conda-forge setuptools=59.5.0 numpy=1.23.1 matplotlib=3.5.3 opencv=4.6.0 pyyaml=6.0 tensorboard=2.10.0 trimesh=3.9.35 configargparse=1.5.3 einops=0.4.1 moviepy=1.0.1 ninja=1.10.2 imageio=2.21.1 pyopengl=3.1.6 gdown=4.5.1
pip install glfw xatlas hydra-core
```

### 2. Install [PyTorch](https://pytorch.org/)
```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
*Note*: The code is tested with PyTorch 1.10.0 and CUDA 11.3.

### 3. Install [NVDiffRec](https://github.com/NVlabs/nvdiffrec) dependencies
```
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch
imageio_download_bin freeimage
```
*Note*: The code is tested with tinycudann=1.6 and it requires GCC/G++ > 7.5 (conda's gxx also works: `conda install -c conda-forge gxx_linux-64=9.4.0`).

### 4. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (for visulaization)
```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

## Data
The preprocessed datasets for each model and the tetrahedral grids files can be downloaded using the scripts in `data/`:

### Tetrahedral Grids
Download the tetrahedral grids:
```
cd data/tets
sh download_tets.sh
```

### MagicPony
Download data used to train the MagicPony, including horses, giraffes, zebras, cows, and birds:
```shell
cd data
sh download_horse_combined.sh
sh download_horse_videos.sh
sh download_giraffe_coco.sh
sh download_zebra_coco.sh
sh download_cow_coco.sh
sh download_bird_videos.sh
```
*Note*: `horse_combined` consists of `horse_videos` from [DOVE](https://dove3d.github.io/) and additional images from [Weizmann Horse Database](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database), [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/), and [Horse-10](http://www.mackenziemathislab.org/horse10).

### 3D-Fauna
Download the preprocessed Fauna datasets :
```shell
cd data/fauna
sh download_fauna_dataset.sh
```
*Note*: `Fauna dataset` consists of manually collected video frames and images from the Internet, and also images from [DOVE](https://dove3d.github.io/), [APT-36K](https://github.com/pandorgan/APT-36K), [Animal3D](https://xujiacong.github.io/Animal3D/), and [Animals-with-Attributes](https://cvml.ista.ac.at/AwA2/).

### Ponymation

```shell
cd data/ponymation
sh download_ponymation_dataset.sh
```

## Pretrained Model

The pretrained models can be downloaded using the scripts in `results/`:

### MagicPony

```shell
cd results/magicpony
sh download_pretrained_magicpony.sh
```
### 3D-Fauna

```shell
cd results/fauna
sh download_pretrained_fauna.sh
```

### Ponymation

```shell
cd results/ponymation
sh download_pretrained_ponymation.sh
```

## Run

### Testing with pretrained checkpoint
After downloading the pretrained checkpoint, we can export results on test images in dataset. The test script saves some visualizations, including reconstruction images from input view, viewpoint and the resulting mesh in `.obj` format. For more visualizations, use `scripts/visualize_results.py` as explained below.

`CONFIG_NAME` can be e.g., `test_magicpony_horse`, `test_ponymation_horse`, `test_fauna`.

```shell
python run.py --config-name CONFIG_NAME
```

### Training
You can also run model training with different config names for different methods.

#### MagicPony

```shell
# Train horse/bird from scratch
python run.py --config-name train_magicpony_horse
```
```shell
# Finetune cow/giraffe/zebra from horse pretrained weights
python run.py --config-name finetune_magicpony_cow
```

#### Ponymation

```shell
# Train horse/cow/giraffe/zebra stage 1
python run.py --config-name train_ponymation_horse_stage1
```

```shell
# Train horse/cow/giraffe/zebra stage 2
python run.py --config-name train_ponymation_horse_stage2
```

#### 3D-Fauna

```shell
python run.py --config-name train_fauna
```

### Visualization

Run and save some visualization results. 

#### MagicPony

```shell
python visualization/visualize_results.py --config-name test_magicpony_horse
```

#### Ponymation

```shell
python visualization/visualize_results.py --config-name test_ponymation_horse
```

#### 3D-Fauna

```shell
python visualization/visualize_results_fauna.py --config-name test_fauna
```

#### Render Modes

Supported `render_modes` include:

- `input_view`: image rendered from the input viewpoint of the reconstructed textured mesh, shading map, gray shape visualization
- `other_views`: image rendered from 12 viewpoints rotating around the vertical axis of the reconstructed textured mesh, gray shape visualization
- `rotation`: video rendered from continuously rotating viewpoints around the vertical axis of the reconstructed textured mesh, gray shape visualization
- `animation` (only supported for horses): two videos rendered from both a side viewpoint and continuously rotating viewpoints of the reconstructed textured mesh animated by interpolating a sequence of pre-configured articulation parameters. `arti_param_dir` can be set to `./visualization/animation_params` which contains a sequence of pre-computed keyframe articulation parameters.
- `canonicalization` (only supported for horses): video of the reconstructed textured mesh morphing from the input pose to a pre-configured canonical pose

#### Test-time Texture Finetuning

To enable test time texture finetuning, set the config `finetune_texture: true`, and (optionally) adjust the number of finetune iterations `finetune_iters` and learning rate `finetune_lr`.

For more precise texture optimization, provide instance masks in the same folder as `*_mask.png`. Otherwise, the background pixels might be pasted onto the object if shape predictions are not perfect aligned.

## Citation

If you use this repository or find the papers useful for your research, please consider citing:
```
@InProceedings{wu2023magicpony,
  author    = {Shangzhe Wu and Ruining Li and Tomas Jakab and Christian Rupprecht and Andrea Vedaldi},
  title     = {{MagicPony}: Learning Articulated 3D Animals in the Wild},
  booktitle = {CVPR},
  year      = {2023}
}
```

```
@InProceedings{Li_2024_CVPR,
    author    = {Li, Zizhang and Litvak, Dor and Li, Ruining and Zhang, Yunzhi and Jakab, Tomas and Rupprecht, Christian and Wu, Shangzhe and Vedaldi, Andrea and Wu, Jiajun},
    title     = {Learning the 3D Fauna of the Web},
    booktitle = {CVPR},
    year      = {2024}
}
```

```
@Article{sun2024ponymation,
  title     = {{Ponymation}: Learning Articulated 3D Animal Motions from Unlabeled Online Videos},
  author    = {Sun, Keqiang and Litvak, Dor and Zhang, Yunzhi and Li, Hongsheng and Wu, Jiajun and Wu, Shangzhe},
  journal   = {ECCV},
  year      = {2024}
}
```

## TODO

- [ ] Ponymation dataset update
- [ ] Data processing script
- [ ] Metrics evaluation script