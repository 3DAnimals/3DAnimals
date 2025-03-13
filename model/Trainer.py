import os
import os.path as osp
from glob import glob
from datetime import datetime
from dataclasses import dataclass
import torch
from tqdm import tqdm
from math import ceil
from accelerate import Accelerator
from .utils import meters, misc
from .dataloaders import DataLoaderConfig, get_data_loaders
from .models import AnimalModelConfig


@dataclass
class TrainerConfig:
    dataset: DataLoaderConfig
    model: AnimalModelConfig

    run_train: bool = True
    run_test: bool = False
    seed: int = 0
    gpu: str = '0'
    num_iters: int = 1
    mixed_precision: str = None

    checkpoint_dir: str = 'results'
    checkpoint_path: str = None
    save_checkpoint_freq: int = 5000
    keep_num_checkpoint: int = 2  # -1 for keeping all checkpoints
    archive_code: bool = True
    resume: bool = True
    checkpoint_name: str = None
    test_result_dir: str = None
    load_optim: bool = True
    reset_epoch: bool = False

    use_logger: bool = True
    logger_type: str = "tensorboard"
    log_image_freq: int = 1000
    log_loss_freq: int = 100
    log_train: bool = True
    log_val: bool = True
    fix_log_batch: bool = False
    save_train_result_freq: int = None

    disc_train: bool = False
    remake_dataloader_iter: int = -1
    remake_dataloader_num: int = 1000
    shuffle_dataset_paths: bool = False


class Trainer:
    def __init__(self, cfg: TrainerConfig, model):
        self.cfg = misc.load_cfg(self, cfg, TrainerConfig)
        misc.setup_runtime(self.cfg)

        if self.cfg.remake_dataloader_iter > 0:
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.cfg.dataset, self.cfg.remake_dataloader_num)
        else:
            self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.cfg.dataset)

        self.current_epoch = 0
        self.metrics_trace = meters.MetricsTrace()
        self.make_metrics = lambda m=None: meters.StandardMetrics(m)
        self.train_result_dir = osp.join(self.checkpoint_dir, 'training_results')
        self.model = model
        self.model.trainer = self
        self.accelerator = Accelerator()
        self.model.accelerator = self.accelerator

        # TODO: use config to define this
        if hasattr(self.model, "all_category_names") and self.model.all_category_names is not None:
            self.model.all_category_names = self.train_loader.dataset.all_category_names

        if self.cfg.remake_dataloader_iter > 0:
            self.remake_dataloader = False

    def load_checkpoint(self):
        """Search the specified/latest checkpoint in checkpoint_dir and load the model and optimizer."""
        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
        elif self.checkpoint_name is not None:
            checkpoint_path = osp.join(self.checkpoint_dir, self.checkpoint_name)
        else:
            checkpoints = sorted(
                glob(osp.join(self.checkpoint_dir, '*.pth')),
                key=lambda x: int(''.join([c for c in osp.basename(x) if c.isdigit()]))
            )
            if len(checkpoints) == 0:
                print(f"No checkpoint found in {self.checkpoint_dir}, train from scratch")
                return 0, 0
            checkpoint_path = checkpoints[-1]
            self.checkpoint_name = osp.basename(checkpoint_path)
        print(f"Loading checkpoint from {checkpoint_path}")
        cp = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_model_state(cp)
        if self.load_optim:
            self.model.load_optimizer_state(cp)
        self.metrics_trace = cp.get('metrics_trace', self.metrics_trace)
        if self.reset_epoch:
            return 0, 0
        else:
            epoch = cp.get('epoch', 999)
            total_iter = cp.get('total_iter', 999999)
            return epoch, total_iter

    def save_checkpoint(self, epoch, total_iter=0, save_optim=True):
        """Save model, optimizer, and metrics state to a checkpoint in checkpoint_dir for the specified iteration."""
        misc.xmkdir(self.checkpoint_dir)
        checkpoint_path = osp.join(self.checkpoint_dir, f'checkpoint{total_iter}.pth')
        state_dict = self.model.get_model_state()
        if save_optim:
            optimizer_state = self.model.get_optimizer_state()
            state_dict = {**state_dict, **optimizer_state}
        state_dict['metrics_trace'] = self.metrics_trace
        state_dict['epoch'] = epoch
        state_dict['total_iter'] = total_iter
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state_dict, checkpoint_path)
        if self.keep_num_checkpoint > 0:
            misc.clean_checkpoint(self.checkpoint_dir, keep_num=self.keep_num_checkpoint)
        self.checkpoint_path = checkpoint_path

    def save_clean_checkpoint(self, path):
        """Save model state only to specified path."""
        torch.save(self.model.get_model_state(), path)

    def test(self):
        """Perform testing."""
        assert self.test_loader is not None, "test_data_dir must be specified for testing"
        self.test_loader = self.accelerator.prepare_data_loader(self.test_loader)
        self.model.to(self.accelerator.device)
        self.model.set_eval()
        self.load_optim = False
        epoch, self.total_iter = self.load_checkpoint()

        if self.test_result_dir is None:
            self.test_result_dir = osp.join(self.checkpoint_dir, f'test_results_{self.checkpoint_name}'.replace('.pth', ''))
        print(f"Saving testing results to {self.test_result_dir}")

        with torch.no_grad():
            for iteration, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                batch = validate_all_to_device(batch, device=self.accelerator.device)
                m = self.model.forward(batch, epoch=epoch, total_iter=self.total_iter, save_results=True, save_dir=self.test_result_dir, is_training=False)
                print(f"T{self.total_iter:06}")

    def train(self):
        """Perform training."""
        assert self.train_loader is not None, "train_data_dir must be specified for training"

        # archive code and configs
        if self.accelerator.is_main_process and self.archive_code:
            misc.archive_code(osp.join(self.checkpoint_dir, 'archived_code.zip'), filetypes=['.py'])

        # initialize
        start_epoch = 0
        self.total_iter = 0
        self.metrics_trace.reset()
        self.model.reset_optimizers()
        self.model.set_train()

        # resume from checkpoint
        if self.resume:
            start_epoch, self.total_iter = self.load_checkpoint()

        self.model.set_train_post_load()

        # setup distributed training
        self.train_loader = self.accelerator.prepare_data_loader(self.train_loader)
        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare_data_loader(self.val_loader)
        for name, value in vars(self.model).items():
            if isinstance(value, torch.nn.Module):
                setattr(self.model, name, self.accelerator.prepare_model(value))
            if isinstance(value, torch.optim.Optimizer):
                setattr(self.model, name, self.accelerator.prepare_optimizer(value))
            if isinstance(value, torch.optim.lr_scheduler._LRScheduler):
                setattr(self.model, name, self.accelerator.prepare_scheduler(value))
        self.model.to(self.accelerator.device)

        # initialize logger
        if self.accelerator.is_main_process and self.use_logger:
            if self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter
                self.logger = SummaryWriter(osp.join(self.checkpoint_dir, 'tensorboard_logs', datetime.now().strftime("%Y%m%d-%H%M%S")), flush_secs=10)
            elif self.logger_type == "wandb":
                from .utils.wandb_writer import WandbWriter
                if self.cfg.dataset.local_dir:
                    try:
                        os.makedirs(self.cfg.dataset.local_dir, exist_ok=True)
                    except Exception:
                        self.cfg.dataset.local_dir = self.cfg.dataset.local_dir.replace("/scr-ssd/", "/scr/")
                        os.makedirs(self.cfg.dataset.local_dir, exist_ok=True)
                self.logger = WandbWriter(project=self.model.name, config=self.cfg, local_dir=self.cfg.dataset.local_dir)
            else:
                raise NotImplementedError(f"Unsupported loger: {self.logger}")
        else:
            self.logger = None

        if self.log_val:
            assert self.val_loader is not None, "val_data_dir must be specified for logging validation"
            self.val_data_iterator = indefinite_generator(self.val_loader)
        if self.fix_log_batch:
            self.log_batch = next(self.val_data_iterator)

        # setup mixed_precision
        if self.mixed_precision:
            if self.mixed_precision == "fp16":
                self.model.mixed_precision = torch.float16
            elif self.mixed_precision == "bf16":
                self.model.mixed_precision = torch.bfloat16
            else:
                raise NotImplementedError(f"Unsupported mixed precision: {self.mixed_precision}")
            # torch.cpu.amp.GradScaler() or torch.GradScaler() need latest version of pytorch
            self.model.scaler = torch.cuda.amp.GradScaler() if "cuda" in str(self.accelerator.device) else None
        else:
            self.model.mixed_precision = None

        # run epochs
        epochs_to_run = ceil((self.num_iters - self.total_iter) / len(self.train_loader))
        for epoch in range(start_epoch, start_epoch + epochs_to_run):
            metrics = self.run_train_epoch(epoch)
            if self.cfg.shuffle_dataset_paths:
                self.train_loader.dataset._shuffle_all()
            self.metrics_trace.append("train", metrics)
            self.metrics_trace.save(osp.join(self.checkpoint_dir, 'metrics.json'))
        print(f"Training completed for all {self.total_iter} iterations.")
        if self.use_logger and self.logger is not None and self.logger_type == "wandb":
            self.logger.finish()

    def run_train_epoch(self, epoch):
        metrics = self.make_metrics()
        for iteration, batch in enumerate(self.train_loader):
            self.total_iter += 1

            if self.cfg.remake_dataloader_iter > 0:
                if not self.remake_dataloader:
                    if self.total_iter >= self.cfg.remake_dataloader_iter:
                        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(self.cfg.dataset)
                        self.model.all_category_names = self.train_loader.dataset.all_category_names
                        self.remake_dataloader = True

            batch = validate_all_to_device(batch, device=self.accelerator.device)
            m = self.model.forward(batch, epoch=epoch, total_iter=self.total_iter, is_training=True)
            self.model.backward()

            if self.cfg.disc_train and (self.model.cfg_mask_discriminator.enable_iter[0] < self.total_iter) and (self.model.cfg_mask_discriminator.enable_iter[1] > self.total_iter):
                # the discriminator training
                discriminator_loss_dict, grad_loss = self.model.discriminator_step()
                m.update(
                    {
                        'mask_disc_loss_discriminator': discriminator_loss_dict['discriminator_loss'] - grad_loss, 
                        'mask_disc_loss_discriminator_grad': grad_loss,
                        'mask_disc_loss_discriminator_rv': discriminator_loss_dict['discriminator_loss_rv'],
                        'mask_disc_loss_discriminator_iv': discriminator_loss_dict['discriminator_loss_iv'],
                        'mask_disc_loss_discriminator_gt': discriminator_loss_dict['discriminator_loss_gt']
                    }
                )
            else:
                discriminator_loss_dict = None

            if self.accelerator.is_main_process:
                num_seqs, num_frames = batch[0].shape[:2]
                total_im_num = num_seqs*num_frames
                metrics.update(m, total_im_num)
                print(f"T{self.total_iter:06}/{metrics}")

            if self.use_logger:
                if self.accelerator.is_main_process and self.total_iter % self.log_loss_freq == 0:
                    self.logger.add_scalar(f"epoch", epoch, self.total_iter)
                    for name, loss in m.items():
                        self.logger.add_scalar(f'train_loss/{name}', loss, self.total_iter)
                    if discriminator_loss_dict is not None:
                        self.logger.add_histogram('train_'+'discriminator_logits/random_view', discriminator_loss_dict['d_rv'], self.total_iter)
                        if discriminator_loss_dict['d_iv'] is not None:
                            self.logger.add_histogram('train_'+'discriminator_logits/input_view', discriminator_loss_dict['d_iv'], self.total_iter)
                        if discriminator_loss_dict['d_gt'] is not None:
                            self.logger.add_histogram('train_'+'discriminator_logits/gt_view', discriminator_loss_dict['d_gt'], self.total_iter)

                if self.save_train_result_freq is not None and self.total_iter % self.save_train_result_freq == 0:
                    with torch.no_grad():
                        m = self.model.forward(batch, epoch=epoch, total_iter=self.total_iter, save_results=True, save_dir=self.train_result_dir, is_training=False)
                        torch.cuda.empty_cache()

                if self.total_iter % self.log_image_freq == 0:
                    if self.log_train:
                        with torch.no_grad():
                            m = self.model.forward(batch, epoch=epoch, logger=self.logger, total_iter=self.total_iter, logger_prefix='train_', is_training=True)
                    if self.log_val:
                        if self.fix_log_batch:
                            batch = self.log_batch
                        else:
                            batch = next(self.val_data_iterator)
                        batch = validate_all_to_device(batch, device=self.accelerator.device)
                        self.model.set_eval()
                        with torch.no_grad():
                            m = self.model.forward(batch, epoch=epoch, logger=self.logger, total_iter=self.total_iter, logger_prefix='val_', is_training=False)
                        self.model.set_train()
                        if self.logger is not None:
                            for name, loss in m.items():
                                self.logger.add_scalar(f'val_loss/{name}', loss, self.total_iter)
                torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            self.model.scheduler_step()
            if self.accelerator.is_main_process and self.total_iter % self.save_checkpoint_freq == 0:
                self.save_checkpoint(epoch+1, total_iter=self.total_iter, save_optim=True)
            self.accelerator.wait_for_everyone()
            if self.total_iter >= self.num_iters:
                break
        return metrics


## utility functions
def indefinite_generator(loader):
    while True:
        for x in loader:
            yield x


def validate_tensor_to_device(x, device=None):
    if isinstance(x, (list, tuple)):
        return x
    if torch.any(torch.isnan(x)):
        return None
    elif device is None:
        return x
    else:
        return x.to(device)

def validate_all_to_device(batch, device=None):
    return tuple(validate_tensor_to_device(x, device) for x in batch)