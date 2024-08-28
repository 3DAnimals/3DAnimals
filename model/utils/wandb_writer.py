import wandb
import numpy as np
from einops import rearrange
from torchvision.transforms.functional import to_pil_image
from dataclasses import is_dataclass, asdict
from omegaconf import OmegaConf


class WandbWriter:
    def __init__(self, project, config=None, **kwargs):
        if OmegaConf.is_config(config):
            config = OmegaConf.to_container(config)
        elif is_dataclass(config):
            config = asdict(config)
        assert isinstance(config, dict)
        self.run = (
            wandb.init(project=project, config=config, **kwargs) if not wandb.run else wandb.run
        )

    def watch(self, models, log_freq, log="all", log_graph=False):
        self.run.watch(models, log=log, log_freq=log_freq, log_graph=log_graph)

    def unwatch(self, models):
        self.run.unwatch(models)

    def add_scalar(self, tag, scalar_value, global_step):
        self.run.log({tag: scalar_value}, step=global_step)

    def add_image(self, tag, img_tensor, global_step=None, dataformats='CHW'):
        dataformats = ' '.join(dataformats)
        img_tensor = rearrange(img_tensor, f"{dataformats} -> C H W")
        image = wandb.Image(to_pil_image(img_tensor), caption=tag)
        self.run.log({tag: image}, step=global_step)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4):
        vid_numpy = (vid_tensor.cpu().numpy()*255).astype(np.uint8)
        video = wandb.Video(vid_numpy, caption=tag, fps=fps)
        self.run.log({tag: video}, step=global_step)

    def add_histogram(self, tag, values, global_step=None):
        self.run.log({tag: wandb.Histogram(values.cpu().numpy())}, step=global_step)

    def finish(self):
        self.run.finish()
