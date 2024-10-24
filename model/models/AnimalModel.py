from dataclasses import dataclass, field
from typing import List, Optional, Dict
from types import SimpleNamespace
import numpy as np
import pytorch3d
import torch
import torch.nn as nn
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt
from einops import rearrange
from model.utils import misc
from model.utils.smooth_loss import SmoothLoss
from model.dataloaders import DataLoaderConfig
from model.render import util
from model.render import render


@dataclass
class OptimizerConfig:
    lr: float = 0.0001
    weight_decay: float = 0.0
    use_scheduler: bool = False
    scheduler_milestone: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    scheduler_gamma: float = 0.5


@dataclass
class RenderConfig:
    spatial_scale: float = 5.
    background_mode: str = 'none'  # none (black), white, checkerboard, background, input
    render_flow: bool = False
    cam_pos_z_offset: float = 10.0
    fov: float = 25.
    renderer_spp: int = 1
    offset_extra: float = 0.0
    render_default: bool = False


@dataclass
class LossConfig:
    mask_loss_weight: float = 10.
    mask_dt_loss_weight: float = 0.
    mask_inv_dt_loss_weight: float = 100.
    rgb_loss_weight: float = 1.
    dino_feat_im_loss_weight: float = 10.
    sdf_bce_reg_loss_weight: float = 0.
    sdf_gradient_reg_loss_weight: float = 0.01
    logit_loss_weight: float = 1.
    logit_loss_target_weight: float = 0.
    logit_loss_dino_feat_im_loss_multiplier: float = 50.  # increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
    arti_reg_loss_iter_range: List[int] = field(default_factory=lambda: [60000, float("inf")])
    arti_reg_loss_weight: float = 0.1
    deform_reg_loss_weight: float = 10.

    smooth_type: str = "dislocation"
    loss_type: str = "l2"
    arti_smooth_loss_weight: float = 0.
    deform_smooth_loss_weight: float = 0.
    campose_smooth_loss_weight: float = 0.
    camposevel_smooth_loss_weight: float = 0.
    artivel_smooth_loss_weight: float = 0.
    bone_smooth_loss_weight: float = 0.
    bonevel_smooth_loss_weight: float = 0.

    arti_recon_loss_weight: float = 5.0
    kld_loss_weight: float = 0.001

    mask_disc_loss_weight: float = 0.1
    mask_disc_loss_rv_weight: float = 0.0
    mask_disc_loss_iv_weight: float = 0.0

    logit_loss_dino_feat_im_loss_multiplier_dict: Dict[int, float] = field(default_factory=lambda: {0: 50., 300000: 500.})
    dino_feat_im_loss_weight_dict: Dict[int, float] =  field(default_factory=lambda: {0: 10., 300000: 1.})
    logit_loss_mask_multiplier: float = 0.05
    logit_loss_mask_inv_dt_multiplier: float = 0.05

@dataclass
class AnimalModelConfig:
    name: str
    dataset: DataLoaderConfig
    cfg_optim_base: OptimizerConfig
    cfg_optim_instance: OptimizerConfig
    cfg_render: RenderConfig
    cfg_loss: LossConfig
    cfg_predictor_base: None
    cfg_predictor_instance: None
    enable_render: bool = True

    extra_renders: Optional[List] = None


class AnimalModel:
    def __init__(self, cfg: AnimalModelConfig):
        misc.load_cfg(self, cfg, AnimalModelConfig)

        if self.cfg_optim_base.use_scheduler:
            self.make_scheduler_base = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.cfg_optim_base.scheduler_milestone, gamma=self.cfg_optim_base.scheduler_gamma)
        if self.cfg_optim_instance.use_scheduler:
            self.make_scheduler_instance = lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.cfg_optim_instance.scheduler_milestone, gamma=self.cfg_optim_instance.scheduler_gamma)

        self.glctx = dr.RasterizeGLContext()
        self.device = None
        self.mixed_precision = False
        self.total_loss = 0.

        self.smooth_loss_fn = SmoothLoss(
            frame_dim=1, smooth_type=self.cfg_loss.smooth_type, loss_type=self.cfg_loss.loss_type
        )

    def load_model_state(self, cp):
        base_missing, base_unexpected = self.netBase.load_state_dict(cp["netBase"], strict=False)
        instance_missing, instance_unexpected = self.netInstance.load_state_dict(cp["netInstance"], strict=False)
        if base_missing: print(f"Missing keys in netBase:\n{base_missing}")
        if base_unexpected: print(f"Unexpected keys in netBase:\n{base_unexpected}")
        if instance_missing: print(f"Missing keys in netInstance:\n{instance_missing}")
        if instance_unexpected: print(f"Unexpected keys in netInstance:\n{instance_unexpected}")

    def load_optimizer_state(self, cp):
        self.optimizerBase.load_state_dict(cp["optimizerBase"])
        self.optimizerInstance.load_state_dict(cp["optimizerInstance"])
        if self.cfg_optim_base.use_scheduler and 'schedulerBase' in cp:
            self.schedulerBase.load_state_dict(cp["schedulerBase"])
        if self.cfg_optim_instance.use_scheduler and 'schedulerInstance' in cp:
            self.schedulerInstance.load_state_dict(cp["schedulerInstance"])

    def get_model_state(self):
        state = {"netBase": self.netBase.state_dict(),
                 "netInstance": self.netInstance.state_dict()}
        return state

    def get_optimizer_state(self):
        state = {"optimizerBase": self.optimizerBase.state_dict(),
                 "optimizerInstance": self.optimizerInstance.state_dict()}
        if self.cfg_optim_base.use_scheduler:
            state["schedulerBase"] = self.schedulerBase.state_dict()
        if self.cfg_optim_instance.use_scheduler:
            state["schedulerInstance"] = self.schedulerInstance.state_dict()
        return state

    def to(self, device):
        self.device = device
        self.netBase.to(device)
        self.netInstance.to(device)

    def set_train(self):
        self.netBase.train()
        self.netInstance.train()

    def set_eval(self):
        self.netBase.eval()
        self.netInstance.eval()

    def reset_optimizers(self):
        print("Resetting optimizers...")
        self.optimizerBase = get_optimizer(self.netBase, lr=self.cfg_optim_base.lr, weight_decay=self.cfg_optim_base.weight_decay)
        self.optimizerInstance = get_optimizer(self.netInstance, self.cfg_optim_instance.lr, weight_decay=self.cfg_optim_instance.weight_decay)
        if self.cfg_optim_base.use_scheduler:
            self.schedulerBase = self.make_scheduler_base(self.optimizerBase)
        if self.cfg_optim_instance.use_scheduler:
            self.schedulerInstance = self.make_scheduler_instance(self.optimizerInstance)

    def backward(self):
        self.optimizerInstance.zero_grad()
        self.optimizerBase.zero_grad()
        if self.mixed_precision and self.scaler:
            self.scaler.scale(self.total_loss).backward()
            # Step optimizer if it is used, Unused optimizers will raise assertion error when unscaling
            try:
                self.scaler.step(self.optimizerInstance)
            except AssertionError:
                pass
            try:
                self.scaler.step(self.optimizerBase)
            except AssertionError:
                pass
            self.scaler.update()
        else:
            self.total_loss.backward()
            self.optimizerInstance.step()
            self.optimizerBase.step()
        self.total_loss = 0.

    def scheduler_step(self):
        if self.cfg_optim_base.use_scheduler:
            self.schedulerBase.step()
        if self.cfg_optim_instance.use_scheduler:
            self.schedulerInstance.step()

    def render(self, render_modes, shape, texture, mvp, w2c, campos, resolution, background=None, im_features=None, light=None, prior_shape=None, dino_net=None, bsdf='diffuse', two_sided_shading=True, num_frames=None, spp=None, class_vector=None):
        mvp, w2c, campos, im_features = map(to_float, [mvp, w2c, campos, im_features])
        h, w = resolution
        N = len(mvp)
        if background is None:
            background = self.cfg_render.background_mode
        if spp is None:
            spp = self.cfg_render.renderer_spp

        if background in ['none', 'black']:
            bg_image = torch.zeros((N, h, w, 3), device=mvp.device)
        elif background == 'white':
            bg_image = torch.ones((N, h, w, 3), device=mvp.device)
        elif background == 'checkerboard':
            bg_image = torch.FloatTensor(util.checkerboard((h, w), 8), device=mvp.device).repeat(N, 1, 1, 1)  # NxHxWxC
        else:
            raise NotImplementedError

        rendered = render.render_mesh(
            self.glctx,
            shape,
            mtx_in=mvp,
            w2c=w2c,
            view_pos=campos,
            material=texture,
            lgt=light,
            resolution=resolution,
            spp=spp,
            num_layers=1,
            msaa=True,
            background=bg_image,
            bsdf=bsdf,
            feat=im_features,
            render_modes=render_modes,
            prior_mesh=prior_shape,
            two_sided_shading=two_sided_shading,
            dino_net=dino_net,
            num_frames=num_frames,
            class_vector=class_vector)
        return rendered

    def compute_reconstruction_losses(self, image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt, dino_feat_im_pred, background_mode='none', reduce=False):
        losses = {}
        batch_size, num_frames, _, h, w = image_pred.shape  # BxFxCxHxW

        ## mask L2 loss
        mask_pred_valid = mask_pred * mask_valid
        mask_loss = (mask_pred_valid - mask_gt) ** 2
        losses['mask_loss'] = mask_loss.view(batch_size, num_frames, -1).mean(2)
        losses['mask_dt_loss'] = (mask_pred * mask_dt[:,:,1]).view(batch_size, num_frames, -1).mean(2)
        losses['mask_inv_dt_loss'] = ((1-mask_pred) * mask_dt[:,:,0]).view(batch_size, num_frames, -1).mean(2)

        mask_pred_binary = (mask_pred_valid > 0.).float().detach()
        mask_both_binary = collapseBF(mask_pred_binary * mask_gt)  # BFxHxW
        mask_both_binary = (nn.functional.avg_pool2d(mask_both_binary.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0.99).float().detach()  # erode by 1 pixel
        mask_both_binary = expandBF(mask_both_binary, b=batch_size, f=num_frames)  # BxFxHxW

        ## RGB L1 loss
        rgb_loss = (image_pred - image_gt).abs()
        if background_mode in ['background', 'input']:
            pass
        else:
            rgb_loss = rgb_loss * mask_both_binary.unsqueeze(2)
        losses['rgb_loss'] = rgb_loss.view(batch_size, num_frames, -1).mean(2)

        ## flow loss between consecutive frames
        if flow_pred is not None:
            flow_loss = (flow_pred - flow_gt) ** 2.
            flow_loss_mask = mask_both_binary[:,:-1].unsqueeze(2).expand_as(flow_gt)

            ## ignore frames where GT flow is too large (likely inaccurate)
            large_flow = (flow_gt.abs() > 0.5).float() * flow_loss_mask
            large_flow = (large_flow.view(batch_size, num_frames-1, -1).sum(2) > 0).float()
            self.large_flow = large_flow

            flow_loss = flow_loss * flow_loss_mask * (1 - large_flow[:,:,None,None,None])
            num_mask_pixels = flow_loss_mask.reshape(batch_size, num_frames-1, -1).sum(2).clamp(min=1)
            losses['flow_loss'] = (flow_loss.reshape(batch_size, num_frames-1, -1).sum(2) / num_mask_pixels)

        ## DINO feature loss
        if dino_feat_im_pred is not None and dino_feat_im_gt is not None:
            dino_feat_loss = (dino_feat_im_pred - dino_feat_im_gt) ** 2
            dino_feat_loss = dino_feat_loss * mask_both_binary.unsqueeze(2)
            losses['dino_feat_im_loss'] = dino_feat_loss.reshape(batch_size, num_frames, -1).mean(2)

        if reduce:
            for k, v in losses.item():
                losses[k] = v.mean()
        return losses

    def compute_regularizers(self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None):
        losses = {}
        aux = {}
        losses.update(self.netBase.netShape.get_sdf_reg_loss())
        if arti_params is not None:
            losses['arti_reg_loss'] = (arti_params ** 2).mean()  # R_art
        if deformation is not None:
            losses['deform_reg_loss'] = (deformation ** 2).mean()  # R_def

        # Smooth losses
        if self.dataset.data_type == "sequence" and self.dataset.num_frames > 1:
            b, f = arti_params.shape[:2]
            if self.cfg_loss.deform_smooth_loss_weight > 0 and deformation is not None:
                losses["deform_smooth_loss"] = self.smooth_loss_fn(expandBF(deformation, b, f))
            if arti_params is not None:
                if self.cfg_loss.arti_smooth_loss_weight > 0:
                    losses["arti_smooth_loss"] = self.smooth_loss_fn(arti_params)
                if self.cfg_loss.artivel_smooth_loss_weight > 0:
                    artivel = arti_params[:, 1:, ...] - arti_params[:, :(f-1), ...]
                    losses["artivel_smooth_loss"] = self.smooth_loss_fn(artivel)  # R_temp
            if pose_raw is not None:
                campose = expandBF(pose_raw, b, f)
                if self.cfg_loss.campose_smooth_loss_weight > 0:
                    losses["campose_smooth_loss"] = self.smooth_loss_fn(campose)
                if self.cfg_loss.camposevel_smooth_loss_weight > 0:
                    camposevel = campose[:, 1:, ...] - campose[:, :(f-1), ...]
                    losses["camposevel_smooth_loss"] = self.smooth_loss_fn(camposevel)
            if posed_bones is not None:
                if self.cfg_loss.bone_smooth_loss_weight > 0:
                    losses["bone_smooth_loss"] = self.smooth_loss_fn(posed_bones)
                if self.cfg_loss.bonevel_smooth_loss_weight > 0:
                    bonevel = posed_bones[:, 1:, ...] - posed_bones[:, :(f-1), ...]
                    losses["bonevel_smooth_loss"] = self.smooth_loss_fn(bonevel)
        return losses, aux

    def forward(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, seq_idx, frame_idx = batch
        global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.dataset.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.dataset.out_image_size
        aux_viz = {}

        dino_feat_im_gt = None if dino_feat_im is None else expandBF(torch.nn.functional.interpolate(collapseBF(dino_feat_im), size=[h, w], mode="bilinear"), batch_size, num_frames)[:, :, :self.cfg_predictor_base.cfg_dino.feature_dim]
        dino_cluster_im_gt = None if dino_cluster_im is None else expandBF(torch.nn.functional.interpolate(collapseBF(dino_cluster_im), size=[h, w], mode="nearest"), batch_size, num_frames)

        ## GT image
        image_gt = input_image
        if self.dataset.out_image_size != self.dataset.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'), batch_size, num_frames)
            if flow_gt is not None:
                flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"), batch_size, num_frames-1)

        ## predict prior shape and DINO
        if in_range(total_iter, self.cfg_predictor_base.cfg_shape.grid_res_coarse_iter_range):
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res_coarse
        else:
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res
        if self.netBase.netShape.grid_res != grid_res:
            self.netBase.netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net = self.netBase(total_iter=total_iter, is_training=is_training)
        else:
            prior_shape, dino_net = self.netBase(total_iter=total_iter, is_training=is_training)

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training)
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training)  # first two dim dimensions already collapsed N=(B*F)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)
        final_losses = {}

        ## render images
        if self.enable_render or not is_training:  # Force render for val and test
            render_flow = self.cfg_render.render_flow and num_frames > 1
            render_modes = ['shaded', 'dino_pred']
            if render_flow:
                render_modes += ['flow']
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.device).type, dtype=self.mixed_precision):
                    renders = self.render(
                        render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                    )
            else:
                if self.cfg_render.render_default:
                    renders = self.render(
                        render_modes, shape, texture, self.default_mvp, self.default_w2c, self.default_campos, (h, w), im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                    )
                else:
                    renders = self.render(
                        render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                    )
            if batch_size * num_frames != renders[0].shape[0]:
                batch_size = int(renders[0].shape[0]/num_frames)
            renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
            if render_flow:
                shaded, dino_feat_im_pred, flow_pred = renders
                flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW
            else:
                shaded, dino_feat_im_pred = renders
                flow_pred = None
            image_pred = shaded[:, :, :3]
            mask_pred = shaded[:, :, 3]

            ## compute reconstruction losses
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.device).type, dtype=self.mixed_precision):
                    losses = self.compute_reconstruction_losses(
                        image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                        dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False
                    )
            else:
                losses = self.compute_reconstruction_losses(
                    image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                    dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False
                )

            ## supervise the rotation logits directly with reconstruction loss
            logit_loss_target = None
            if losses is not None:
                logit_loss_target = torch.zeros_like(expandBF(rot_logit, batch_size, num_frames))
                for name, loss in losses.items():
                    loss_weight = getattr(self.cfg_loss, f"{name}_weight")
                    if name in ['dino_feat_im_loss']:
                        ## increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier
                    if loss_weight > 0:
                        logit_loss_target += loss * loss_weight

                    ## multiply the loss with probability of the rotation hypothesis (detached)
                    if self.netInstance.cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
                        loss_prob = rot_prob.detach().view(batch_size, num_frames)[:, :loss.shape[1]]  # handle edge case for flow loss with one frame less
                        loss = loss * loss_prob *self.netInstance.num_pose_hypos
                    ## only compute flow loss for frames with the same rotation hypothesis
                    if name == 'flow_loss' and num_frames > 1:
                        ri = rot_idx.view(batch_size, num_frames)
                        same_rot_idx = (ri[:, 1:] == ri[:, :-1]).float()
                        loss = loss * same_rot_idx
                    ## update the final prob-adjusted losses
                    final_losses[name] = loss.mean()

                logit_loss_target = collapseBF(logit_loss_target).detach()  # detach the gradient for the loss target
                final_losses['logit_loss'] = ((rot_logit - logit_loss_target)**2.).mean()
                final_losses['logit_loss_target'] = logit_loss_target.mean()

        ## regularizers
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.device).type, dtype=self.mixed_precision):
                regularizers, aux = self.compute_regularizers(
                    arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                    posed_bones=forward_aux.get("posed_bones")
                )
        else:
            regularizers, aux = self.compute_regularizers(
                arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                posed_bones=forward_aux.get("posed_bones")
            )
        final_losses.update(regularizers)
        aux_viz.update(aux)

        ## compute final losses
        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = getattr(self.cfg_loss, f"{name}_weight")
            if loss_weight <= 0:
                continue
            if not in_range(total_iter, self.cfg_predictor_instance.cfg_texture.texture_iter_range) and (name in ['rgb_loss']):
                continue
            if not in_range(total_iter, self.cfg_loss.arti_reg_loss_iter_range) and (name in ['arti_reg_loss']):
                continue
            if name in ["logit_loss_target"]:
                continue
            total_loss += loss * loss_weight
        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import pdb; pdb.set_trace()

        metrics = {'loss': total_loss, **final_losses}

        log = SimpleNamespace(**locals())
        if logger is not None and (self.enable_render or not is_training):
            self.log_visuals(log, logger)
        if save_results:
            self.save_results(log)
        return metrics

    def log_visuals(self, log, logger, sdf_feats=None, text=None):
        b0 = max(min(log.batch_size, 16//log.num_frames), 1)
        def log_image(name, image):
            logger.add_image(log.logger_prefix+'image/'+name, misc.image_grid(collapseBF(image[:b0,:]).detach().cpu().clamp(0,1)), log.total_iter)
        def log_video(name, frames):
            logger.add_video(log.logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), log.total_iter, fps=2)

        log_image('image_gt', log.input_image)
        log_image('image_pred', log.image_pred)
        log_image('mask_gt', log.mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
        log_image('mask_pred', log.mask_pred.unsqueeze(2).repeat(1,1,3,1,1))

        if log.dino_feat_im_gt is not None:
            log_image('dino_feat_im_gt', log.dino_feat_im_gt[:,:,:3])
        if log.dino_feat_im_pred is not None:
            log_image('dino_feat_im_pred', log.dino_feat_im_pred[:,:,:3])
        if log.dino_cluster_im_gt is not None:
            log_image('dino_cluster_im_gt', log.dino_cluster_im_gt)

        if self.cfg_render.render_flow and log.flow_gt is not None:
            flow_gt_viz = torch.nn.functional.pad(log.flow_gt, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
            flow_gt_viz = flow_gt_viz + 0.5  # -0.5~1.5
            flow_gt_viz = torch.nn.functional.pad(flow_gt_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization

            ## draw marker on large flow frames
            large_flow_marker_mask = torch.zeros_like(flow_gt_viz)
            large_flow_marker_mask[:,:,:,:8,:8] = 1.
            large_flow = torch.cat([self.large_flow, self.large_flow[:,:1] *0.], 1)
            large_flow_marker_mask = large_flow_marker_mask * large_flow[:,:,None,None,None]
            red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(flow_gt_viz.device)
            flow_gt_viz = large_flow_marker_mask * red + (1-large_flow_marker_mask) * flow_gt_viz
            log_image('flow_gt', flow_gt_viz)

        if self.cfg_render.render_flow and log.flow_pred is not None:
            flow_pred_viz = torch.nn.functional.pad(log.flow_pred, pad=[0, 0, 0, 0, 0, 1])  # add a dummy channel for visualization
            flow_pred_viz = flow_pred_viz + 0.5  # -0.5~1.5
            flow_pred_viz = torch.nn.functional.pad(flow_pred_viz, pad=[0, 0, 0, 0, 0, 0, 0, 1])  # add a dummy frame for visualization
            log_image('flow_pred', flow_pred_viz)

        if log.arti_params is not None:
            logger.add_histogram(log.logger_prefix+'arti_params', log.arti_params, log.total_iter)

        if log.deformation is not None:
            logger.add_histogram(log.logger_prefix+'deformation', log.deformation, log.total_iter)

        rot_rep = self.cfg_predictor_instance.cfg_pose.rot_rep
        if rot_rep == 'euler_angle':
            for i, name in enumerate(['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose[...,i], log.total_iter)
        elif rot_rep == 'quaternion':
            for i, name in enumerate(['qt_0', 'qt_1', 'qt_2', 'qt_3', 'trans_x', 'trans_y', 'trans_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose[...,i], log.total_iter)
            rot_euler = pytorch3d.transforms.matrix_to_euler_angles(pytorch3d.transforms.quaternion_to_matrix(log.pose.detach().cpu()[...,:4]), convention='XYZ')
            for i, name in enumerate(['rot_x', 'rot_y', 'rot_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, rot_euler[...,i], log.total_iter)
        elif rot_rep in ['lookat', 'quadlookat', 'octlookat']:
            for i, name in enumerate(['fwd_x', 'fwd_y', 'fwd_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose_raw[...,i], log.total_iter)
            for i, name in enumerate(['trans_x', 'trans_y', 'trans_z']):
                logger.add_histogram(log.logger_prefix+'pose/'+name, log.pose_raw[...,-3+i], log.total_iter)

        if rot_rep in ['quadlookat', 'octlookat']:
            for i, rp in enumerate(log.forward_aux['rots_probs'].unbind(-1)):
                logger.add_histogram(log.logger_prefix+'pose/rot_prob_%d'%i, rp, log.total_iter)

        if sdf_feats is None:
            logger.add_histogram(log.logger_prefix+'sdf', self.netBase.netShape.get_sdf(), log.total_iter)
        else:
            logger.add_histogram(log.logger_prefix+'sdf', self.netBase.netShape.get_sdf(feats=log.class_vector), log.total_iter)
        logger.add_histogram(log.logger_prefix+'coordinates', log.shape.v_pos, log.total_iter)

        render_modes = ['geo_normal', 'kd', 'shading']
        rendered = self.render(render_modes, log.shape, log.texture, log.mvp, log.w2c, log.campos, (log.h, log.w), im_features=log.im_features, light=log.light, prior_shape=log.prior_shape)
        geo_normal, albedo, shading = map(lambda x: expandBF(x, log.batch_size, log.num_frames), rendered)
        if hasattr(self.netInstance, "articulated_shape_gt"):
            rendered_gt = self.render(render_modes, self.netInstance.articulated_shape_gt, log.texture, log.mvp, log.w2c, log.campos, (log.h, log.w), im_features=log.im_features, light=log.light, prior_shape=log.prior_shape)
            geo_normal_gt, albedo_gt, shading_gt = map(lambda x: expandBF(x, log.batch_size, log.num_frames), rendered_gt)
            del self.netInstance.articulated_shape_gt

        if log.light is not None:
            param_names = ['dir_x', 'dir_y', 'dir_z', 'int_ambient', 'int_diffuse']
            for name, param in zip(param_names, log.light.light_params.unbind(-1)):
                logger.add_histogram(log.logger_prefix+'light/'+name, param, log.total_iter)
            log_image('albedo', albedo)
            log_image('shading', shading.repeat(1,1,3,1,1) /2.)

        ## add bone visualizations
        if 'posed_bones' in log.aux_viz:
            rendered_bone_image = self.render_bones(log.mvp, log.aux_viz['posed_bones'], (log.h, log.w))
            rendered_bone_image_mask = (rendered_bone_image < 1).float()
            geo_normal = rendered_bone_image_mask*0.8 * rendered_bone_image + (1-rendered_bone_image_mask*0.8) * geo_normal
            if log.aux_viz.get("posed_bones_gt") is not None:
                rendered_bone_image = self.render_bones(log.mvp, log.aux_viz['posed_bones_gt'], (log.h, log.w))
                rendered_bone_image_mask = (rendered_bone_image < 1).float()
                geo_normal_gt = rendered_bone_image_mask * 0.8 * rendered_bone_image + (
                            1 - rendered_bone_image_mask * 0.8) * geo_normal_gt

        ## draw marker on images with randomly sampled pose
        if rot_rep in ['quadlookat', 'octlookat']:
            rand_pose_flag = log.forward_aux['rand_pose_flag']
            rand_pose_marker_mask = torch.zeros_like(geo_normal)
            rand_pose_marker_mask[:,:,:,:16,:16] = 1.
            rand_pose_marker_mask = rand_pose_marker_mask * rand_pose_flag.view(log.batch_size, log.num_frames, 1, 1, 1)
            red = torch.FloatTensor([1,0,0]).view(1,1,3,1,1).to(geo_normal.device)
            geo_normal = rand_pose_marker_mask * red + (1-rand_pose_marker_mask) * geo_normal

        log_image('instance_geo_normal', geo_normal)

        rot_frames = self.render_rotation_frames('geo_normal', log.shape, log.texture, log.light, (log.h, log.w), im_features=log.im_features, prior_shape=log.prior_shape, num_frames=15, b=1, text=text)
        log_video('instance_normal_rotation', rot_frames)

        rot_frames = self.render_rotation_frames('shaded', log.prior_shape, log.texture, log.light, (log.h, log.w), im_features=log.im_features, num_frames=15, b=1, text=text)
        log_video('prior_image_rotation', rot_frames)

        rot_frames = self.render_rotation_frames('geo_normal', log.prior_shape, log.texture, log.light, (log.h, log.w), im_features=log.im_features, num_frames=15, b=1, text=text)
        log_video('prior_normal_rotation', rot_frames)

        log.__dict__.update({k: v for k, v in locals().items() if k != "log"})
        return log

    def save_results(self, log):
        b0 = log.batch_size * log.num_frames
        fnames = [f'{log.total_iter:07d}_{fid:10d}' for fid in collapseBF(log.global_frame_id.int())][:b0]
        def save_image(name, image):
            misc.save_images(log.save_dir, collapseBF(image)[:b0].clamp(0,1).detach().cpu().numpy(), suffix=name, fnames=fnames)

        save_image('image_gt', log.image_gt)
        save_image('image_pred', log.image_pred)
        save_image('mask_gt', log.mask_gt.unsqueeze(2).repeat(1,1,3,1,1))
        save_image('mask_pred', log.mask_pred.unsqueeze(2).repeat(1,1,3,1,1))

        if self.cfg_render.render_flow and log.flow_gt is not None:
            flow_gt_viz = torch.cat([log.flow_gt, torch.zeros_like(log.flow_gt[:,:,:1])], 2) + 0.5  # -0.5~1.5
            flow_gt_viz = flow_gt_viz.view(-1, *flow_gt_viz.shape[2:])
            save_image('flow_gt', flow_gt_viz)
        if log.flow_pred is not None:
            flow_pred_viz = torch.cat([log.flow_pred, torch.zeros_like(log.flow_pred[:,:,:1])], 2) + 0.5  # -0.5~1.5
            flow_pred_viz = flow_pred_viz.view(-1, *flow_pred_viz.shape[2:])
            save_image('flow_pred', flow_pred_viz)

        tmp_shape = log.shape.first_n(b0).clone()
        tmp_shape.material = log.texture
        feat = log.im_features[:b0] if log.im_features is not None else None
        misc.save_obj(log.save_dir, tmp_shape, save_material=False, feat=feat, suffix="mesh", fnames=fnames)
        misc.save_txt(log.save_dir, log.pose[:b0].detach().cpu().numpy(), suffix='pose', fnames=fnames)


    def render_rotation_frames(self, render_mode, mesh, texture, light, resolution, background=None, im_features=None, prior_shape=None, num_frames=36, b=None, text=None):
        if b is None:
            b = len(mesh)
        else:
            mesh = mesh.first_n(b)
            feat = im_features[:b] if im_features is not None else None

        delta_angle = np.pi / num_frames * 2
        delta_rot_matrix = torch.FloatTensor([
            [np.cos(delta_angle),  0, np.sin(delta_angle), 0],
            [0,                    1, 0,                   0],
            [-np.sin(delta_angle), 0, np.cos(delta_angle), 0],
            [0,                    0, 0,                   1],
        ]).to(self.device).repeat(b, 1, 1)

        w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
        w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cfg_render.cam_pos_z_offset *1.1])
        w2c = w2c.repeat(b, 1, 1).to(self.device)
        proj = util.perspective(self.cfg_render.fov / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.device)
        mvp = torch.bmm(proj, w2c)
        campos = -w2c[:, :3, 3]

        def rotate_pose(mvp, campos):
            mvp = torch.matmul(mvp, delta_rot_matrix)
            campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]
            return mvp, campos

        frames = []
        for _ in range(num_frames):
            rendered = self.render([render_mode], mesh, texture, mvp, w2c, campos, resolution, background=background, im_features=feat, light=light, prior_shape=prior_shape)
            shaded = rendered[0]
            frames += [misc.image_grid(shaded[:, :3])]
            mvp, campos = rotate_pose(mvp, campos)
        
        if text is not None:
            frames = [torch.Tensor(misc.add_text_to_image(f, text)).permute(2, 0, 1) for f in frames]
        return torch.stack(frames, dim=0)  # Shape: (T, C, H, W)

    def render_bones(self, mvp, bones_pred, size=(256, 256)):
        bone_world4 = torch.concat([bones_pred, torch.ones_like(bones_pred[..., :1]).to(bones_pred.device)], dim=-1)
        b, f, num_bones = bone_world4.shape[:3]
        bones_clip4 = (bone_world4.view(b, f, num_bones*2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)).view(b, f, num_bones, 2, 4)
        bones_uv = bones_clip4[..., :2] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 2
        dpi = 32
        fx, fy = size[1] // dpi, size[0] // dpi

        rendered = []
        for b_idx in range(b):
            for f_idx in range(f):
                frame_bones_uv = bones_uv[b_idx, f_idx].cpu().numpy()
                fig = plt.figure(figsize=(fx, fy), dpi=dpi, frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                for bone in frame_bones_uv:
                    ax.plot(bone[:, 0], bone[:, 1], marker='o', linewidth=8, markersize=20)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.invert_yaxis()
                # Convert to image
                fig.add_axes(ax)
                fig.canvas.draw_idle()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                w, h = fig.canvas.get_width_height()
                image.resize(h, w, 3)
                rendered += [image / 255.]
                plt.close(fig)

        rendered = expandBF(torch.FloatTensor(np.stack(rendered, 0)).permute(0, 3, 1, 2).to(bones_pred.device), b, f)
        return rendered


## utility functions
def in_range(x, range):
    return misc.in_range(x, range, default_indicator=-1)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')


def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def get_optimizer(model, lr=0.0001, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

def to_float(x):
    try:
        return x.float()
    except AttributeError:
        return x
