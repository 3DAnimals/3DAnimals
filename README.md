# 3D Animals Codebase

This repository contains the unified codebase for several projects on articulated 3D animal reconstruction and motion generation, including:

- [MagicPony: Learning Articulated 3D Animals in the Wild](https://3dmagicpony.github.io/) (CVPR 2023) [![arXiv](https://img.shields.io/badge/arXiv-2211.12497-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2211.12497) - a category-specific single-image 3D animal reconstruction model
- [Learning the 3D Fauna of the Web](https://kyleleey.github.io/3DFauna/) (CVPR 2024) [![arXiv](https://img.shields.io/badge/arXiv-2401.02400-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.02400) - a pan-category single-image 3D animal reconstruction model
- [Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos](https://keqiangsun.github.io/projects/ponymation/) (ECCV 2024) [![arXiv](https://img.shields.io/badge/arXiv-2312.13604-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.13604) - an articulated 3D animal motion generative model


## Installation
See [INSTALL.md](./INSTALL.md).

## Data
### Tetrahedral Grids
We adopt the hybrid SDF-mesh representation from [DMTet](https://research.nvidia.com/labs/toronto-ai/DMTet/) to represent the 3D shape of the animals. It uses tetrahedral grids to extract meshes from underlying SDF representation.

Download the pre-computed tetrahedral grids:
```shell
cd data/tets
sh download_tets.sh
```

### Datasets
Download the preprocessed datasets for each project using the download scripts provided in `data/`, for example:
```shell
cd data/magicpony
sh download_horse_combined.sh
```
See the notes [below](#data-1) for the details of each dataset.


## Pretrained Models
The pretrained models can be downloaded using the scripts provided in `results/`, for example:
```shell
cd results/magicpony
sh download_pretrained_horse.sh
```


## Run
Once the data is prepared, both training and inference of all models can be executed using a single command:
```shell
python run.py --config-name CONFIG_NAME
```
`CONFIG_NAME` can be any of the configs specified in `config/`, e.g., `test_magicpony_horse` or `train_magicpony_horse`.

### Testing using the Pretrained Models
The simplest use case is to test the pretrained models on test images. To do this, use the configs in `configs/` that start with `test_*`. Open the config files to check the details, including the path of the test images.

Note that only the RGB images are required during testing. The DINO features are not required. The mask images are only required if you wish to finetune the texture with higher precision for visualization (see [below](#test-time-texture-finetuning)).

When running the command with the default test configs, it will automatically save some basic visualizations, including the reconstructed views and 3D meshes. For more advanced and customized visualizations, use `scripts/visualize_results.py` as explained [below](#visualization).

### Training
See the instructions for each specific model [below](#magicpony-arxiv).

### Visualization
We provide some scripts that we used to generate the visualizations on our project pages ([MagicPony](https://3dmagicpony.github.io/), [3D-Fauna](https://kyleleey.github.io/3DFauna/), [Ponymation](https://keqiangsun.github.io/projects/ponymation/)). To render such visualizations, simply run the following command with the proper test config, e.g.:
```shell
python visualization/visualize_results.py --config-name test_magicpony_horse
```

For 3D-Fauna, use `visualize_results_fauna.py` instead:
```shell
python visualization/visualize_results_fauna.py --config-name test_fauna
```

Check the `#Visualization` section in the config files for specific visualization configurations.

#### Rendering Modes
The visualization script supports the following `render_modes`, which can be specified in the config:
- `input_view`: image rendered from the input viewpoint of the reconstructed textured mesh, shading map, gray shape visualization.
- `other_views`: image rendered from 12 viewpoints rotating around the vertical axis of the reconstructed textured mesh, gray shape visualization.
- `rotation`: video rendered from continuously rotating viewpoints around the vertical axis of the reconstructed textured mesh, gray shape visualization.
- `animation` (only supported for quadrupeds): two videos rendered from both a side viewpoint and continuously rotating viewpoints of the reconstructed textured mesh animated by interpolating a sequence of pre-configured articulation parameters. `arti_param_dir` can be set to `./visualization/animation_params` which contains a sequence of pre-computed keyframe articulation parameters.
- `canonicalization` (only supported for quadrupeds): video of the reconstructed textured mesh morphing from the input pose to a pre-configured canonical pose.

#### Test-time Texture Finetuning
To enable texture finetuning at test time, set `finetune_texture: true` in the config, and (optionally) adjust the number of finetune iterations `finetune_iters` and learning rate `finetune_lr`.

For more precise texture optimization, provide instance masks in the same folder as `*_mask.png`. Otherwise, the background pixels might be pasted onto the object if shape predictions are not perfectly aligned.


## MagicPony [![arXiv](https://img.shields.io/badge/arXiv-2211.12497-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2211.12497)
[MagicPony](https://3dmagicpony.github.io/) learns a category-specific model for single-image articulated 3D reconstruction of an animal species.

### Data
We trained MagicPony models on image collections of horses, giraffes, zebras, cows, and birds. The data download scripts in `data/magicpony` provide access to the following preprocessed datasets:
- `horse_videos` and `bird_videos` were released by [DOVE](https://dove3d.github.io/).
- `horse_combined` consists of `horse_videos` and additional images selected from [Weizmann Horse Database](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database), [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/), and [Horse-10](http://www.mackenziemathislab.org/horse10).
- `giraffe_coco`, `zebra_coco` and `cow_coco` are filtered subsets of the [COCO dataset](https://cocodataset.org/).

### Training
To train MagicPony on the provided horse dataset or bird dataset from scratch, simply use the training configs: `train_magicpony_horse` or `train_magicpony_bird`, e.g.:
```shell
python run.py --config-name train_magicpony_horse
```

To train it on the provided giraffe, zebra, or cow datasets, which are much smaller, please finetune from a _pretrained_ horse model using the finetuning configs: `finetune_magicpony_giraffe`, `finetune_magicpony_zebra`, or `finetune_magicpony_cow`.


## 3D-Fauna [![arXiv](https://img.shields.io/badge/arXiv-2401.02400-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2401.02400)
[3D-Fauna](https://kyleleey.github.io/3DFauna/) learns a pan-category model for single-image articulated 3D reconstruction of any quadruped species.

### Data
The `Fauna Dataset`, which can be downloaded via the script `data/fauna/download_fauna_dataset.sh`, consists of video frames and images sourced from the Internet, as well as images from [DOVE](https://dove3d.github.io/), [APT-36K](https://github.com/pandorgan/APT-36K), [Animal3D](https://xujiacong.github.io/Animal3D/), and [Animals-with-Attributes](https://cvml.ista.ac.at/AwA2/).

### Training
To train 3D-Fauna on the Fauna Dataset, simply run:
```shell
python run.py --config-name train_fauna
```


## Ponymation [![arXiv](https://img.shields.io/badge/arXiv-2312.13604-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2312.13604)
[Ponymation](https://keqiangsun.github.io/projects/ponymation/) learns a generative model of articulated 3D motions of an animal species.

### Data
To be updated shortly.

### Training
Ponymation is trained in two stages. In the first stage, we pretrain a 3D reconstruction model that takes in a sequence of frames and reconstructs a sequence of articulated 3D shapes of the animal. This stage can be initiated using the stage 1 config `train_ponymation_horse_stage1`:
```shell
python run.py --config-name train_ponymation_horse_stage1
```

After this video reconstruction model is pretrained, we then train a generative model of the articulated 3D motions in the second stage, using the stage 2 config `train_ponymation_horse_stage2`:
```shell
python run.py --config-name train_ponymation_horse_stage2
```


## Citation
If you use this repository or find the papers useful for your research, please consider citing the following publications, as well as the original publications of the datasets used:
```
@InProceedings{wu2023magicpony,
  title     = {{MagicPony}: Learning Articulated 3D Animals in the Wild},
  author    = {Wu, Shangzhe and Li, Ruining and Jakab, Tomas and Rupprecht, Christian and Vedaldi, Andrea},
  booktitle = {CVPR},
  year      = {2023}
}
```

```
@InProceedings{li2024fauna,
  title     = {Learning the 3D Fauna of the Web},
  author    = {Li, Zizhang and Litvak, Dor and Li, Ruining and Zhang, Yunzhi and Jakab, Tomas and Rupprecht, Christian and Wu, Shangzhe and Vedaldi, Andrea and Wu, Jiajun},
  booktitle = {CVPR},
  year      = {2024}
}
```

```
@InProceedings{sun2024ponymation,
  title     = {{Ponymation}: Learning Articulated 3D Animal Motions from Unlabeled Online Videos},
  author    = {Sun, Keqiang and Litvak, Dor and Zhang, Yunzhi and Li, Hongsheng and Wu, Jiajun and Wu, Shangzhe},
  booktitle = {ECCV},
  year      = {2024}
}
```

## TODO

- [ ] Ponymation dataset update
- [ ] Data processing script
- [ ] Metrics evaluation script