import torch
import numpy as np
from einops import repeat
from dataclasses import dataclass
from .AnimalModel import AnimalModel, AnimalModelConfig
from ..geometry.skinning import euler_angles_to_matrix
from ..utils import misc
from ..predictors import (
    BasePredictorBase, InstancePredictorMotionVAE, BasePredictorConfig, InstancePredictorMotionVAEConfig
)

@dataclass
class PonymationConfig(AnimalModelConfig):
    cfg_predictor_base: BasePredictorConfig = None
    cfg_predictor_instance: InstancePredictorMotionVAEConfig = None


class Ponymation(AnimalModel):
    def __init__(self, cfg: PonymationConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, PonymationConfig)
        self.netBase = BasePredictorBase(self.cfg_predictor_base)
        self.netInstance = InstancePredictorMotionVAE(self.cfg_predictor_instance)
        self.arti_recon_loss_fn = torch.nn.MSELoss()
        self.get_default_pose()

    def get_default_pose(self):
        
        pose_canon = torch.concat([torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(self.device)
        mvp_canon, w2c_canon, campos_canon = self.netInstance.get_camera_extrinsics_from_pose(pose_canon, offset_extra=self.cfg_render.offset_extra)
        viewpoint_arti = torch.FloatTensor([0, -120, 0]) / 180 * np.pi
        mtx = torch.eye(4).to(self.device)
        mtx[:3, :3] = euler_angles_to_matrix(viewpoint_arti, "XYZ")
        w2c_arti = torch.matmul(w2c_canon, mtx[None])
        mvp_arti = torch.matmul(mvp_canon, mtx[None])
        campos_arti = campos_canon @ torch.linalg.inv(mtx[:3, :3]).T
        self.default_pose, self.default_mvp, self.default_w2c, self.default_campos = pose_canon, mvp_arti, w2c_arti, campos_arti

    def set_eval(self):
        super().set_eval()
        self.netInstance.forward_train = self.netInstance.forward
        if self.netInstance.enable_motion_vae:
            self.netInstance.forward = self.netInstance.generate

    def set_train(self):
        super().set_train()
        if self.netInstance.forward == self.netInstance.generate:
            assert hasattr(self.netInstance, "forward_train")
            self.netInstance.forward = self.netInstance.forward_train

    def load_model_state(self, cp):
        super().load_model_state(cp)
        if self.netInstance.enable_motion_vae:
            for param in self.netBase.parameters():
                param.requires_grad = False
            for param in self.netInstance.parameters():
                param.requires_grad = False
            for param in self.netInstance.netVAE.parameters():
                param.requires_grad = True

    def compute_regularizers(self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None):
        losses, aux = super().compute_regularizers(arti_params, deformation, pose_raw, posed_bones)
        if self.cfg_loss.arti_recon_loss_weight > 0 and hasattr(self.netInstance, "articulation_angles_gt"):
            assert not self.netInstance.articulation_angles_gt.requires_grad
            losses["arti_recon_loss"] = self.arti_recon_loss_fn(  # L_teacher
                self.netInstance.articulation_angles_pred, self.netInstance.articulation_angles_gt
            )
        if (
            self.cfg_loss.kld_loss_weight > 0
            and hasattr(self.netInstance, "log_var_vae") and hasattr(self.netInstance, "mu_vae")
        ):
            losses["kld_loss"] = -0.5 * torch.mean(  # L_KL
                torch.sum(
                    1 + self.netInstance.log_var_vae - self.netInstance.mu_vae ** 2 - self.netInstance.log_var_vae.exp(),
                    dim=1
                )
            )
        return losses, aux

    def compute_reconstruction_losses(
        self, image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
        dino_feat_im_pred, background_mode='none', reduce=False
    ):  # Disable reconstruction loss in motion vae stage
        if self.netInstance.enable_motion_vae:
            return None
        else:
            return super().compute_reconstruction_losses(
                image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                dino_feat_im_pred, background_mode, reduce
            )

    def log_visuals(self, log, logger):
        log = super().log_visuals(log, logger)
        def log_video(name, frames):
            logger.add_video(log.logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), log.total_iter, fps=2)
        if log.num_frames > 1:
            log_video("sequence_image_gt", log.input_image[0])
            log_video("sequence_mask_gt", repeat(log.mask_gt[0], "f h w -> f c h w", c=3))
            if self.netInstance.enable_motion_vae and not log.is_training:
                suffix = "gen"
            else:
                suffix = "pred"
            log_video(f"sequence_image_{suffix}", log.image_pred[0])
            log_video(f"sequence_mask_{suffix}", repeat(log.mask_pred[0], "f h w -> f c h w", c=3))
            log_video(f"sequence_instance_geo_normal_{suffix}", log.geo_normal[0])
            if hasattr(log, "geo_normal_gt"):
                log_video(f"sequence_instance_geo_normal_gt", log.geo_normal_gt[0])
