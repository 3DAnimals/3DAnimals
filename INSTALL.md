## Installation

### 1. Install dependencies
```shell
conda env create -f environment.yml
```
or manually:
```shell
conda install -c conda-forge setuptools=59.5.0 numpy=1.23.1 matplotlib=3.5.3 opencv=4.6.0 pyyaml=6.0 tensorboard=2.10.0 trimesh=3.9.35 configargparse=1.5.3 einops=0.4.1 moviepy=1.0.1 ninja=1.10.2 imageio=2.21.1 pyopengl=3.1.6 gdown=4.5.1 hydra-core=1.3.2
pip install glfw xatlas
```

### 2. Install [PyTorch](https://pytorch.org/)
```shell
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
*Note*: The code is tested with PyTorch 1.10.0 and CUDA 11.3.

### 3. Install [NVDiffRec](https://github.com/NVlabs/nvdiffrec) dependencies
```shell
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn@v1.6#subdirectory=bindings/torch
imageio_download_bin freeimage
```
*Note*: The code is tested with tinycudann=1.6. It requires GCC/G++ > 7.5 (conda's gxx also works: `conda install -c conda-forge gxx_linux-64=9.4.0`). If you run into error `libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`, try `conda install -c conda-forge mkl==2024.0` according to [avivko](https://github.com/pytorch/pytorch/issues/123097#issuecomment-2105963891).

### 4. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) (for visulaization)
```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```