from dataclasses import dataclass, field
from collections import OrderedDict
from typing import List, Dict
from types import SimpleNamespace
import numpy as np
import torch
import pytorch3d
from model.render import util
from model.render import render
from .AnimalModel import AnimalModel, AnimalModelConfig, OptimizerConfig, get_optimizer, to_float, expandBF, collapseBF, in_range
from ..utils import misc
from ..predictors import BasePredictorBank, InstancePredictorBase, BasePredictorBankConfig, InstancePredictorConfig, FaunaInstancePredictorConfig, InstancePredictorFauna
from ..networks import discriminator_architecture


@dataclass
class MaskDiscriminatorConfig:
    enable_iter: List[int] = field(default_factory=lambda: [80000, 300000])
    disc_gt: bool = False
    disc_iv: bool = True
    disc_iv_label: str = 'Real'
    mask_disc_loss_weight: float = 0.1
    discriminator_loss_weight: float = 1.0
    disc_reg_mul: float = 10.


@dataclass
class FaunaConfig(AnimalModelConfig):
    cfg_predictor_base: BasePredictorBankConfig = None
    cfg_predictor_instance: FaunaInstancePredictorConfig = None
    cfg_mask_discriminator: MaskDiscriminatorConfig = None
    cfg_optim_discriminator: OptimizerConfig = None


class FaunaModel(AnimalModel):
    def __init__(self, cfg: FaunaConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, FaunaConfig)
        self.netBase = BasePredictorBank(self.cfg_predictor_base)
        self.netInstance = InstancePredictorFauna(self.cfg_predictor_instance)

        # this module is not in netInstance or netBase
        self.netDisc = discriminator_architecture.DCDiscriminator(in_dim=(self.netBase.cfg_bank.memory_bank_dim + 1))
        self.record_mask_gt = None
        self.record_mask_iv = None
        self.record_mask_rv = None
        self.discriminator_loss = 0.

        self.all_category_names = None
    
    def load_model_state(self, cp):
        super().load_model_state(cp=cp)
        disc_missing, disc_unexpected = self.netDisc.load_state_dict(cp["netDisc"], strict=False)
        if disc_missing: print(f"Missing keys in netBase:\n{disc_missing}")
        if disc_unexpected: print(f"Unexpected keys in netBase:\n{disc_unexpected}")
    
    def load_optimizer_state(self, cp):
        super().load_optimizer_state(cp)
        self.optimizerDisc.load_state_dict(cp["optimizerDisc"])

    def get_model_state(self):
        state = super().get_model_state()
        state.update({"netDisc": self.accelerator.unwrap_model(self.netDisc).state_dict()})
        return state

    def get_optimizer_state(self):
        state = super().get_optimizer_state()
        state.update({"optimizerDisc": self.optimizerDisc.state_dict()})
        return state

    def to(self, device):
        super().to(device)
        self.get_predictor("netDisc").to(device)

    def set_train(self):
        super().set_train()
        self.get_predictor("netDisc").train()

    def set_eval(self):
        super().set_eval()
        self.get_predictor("netDisc").eval()

    def reset_optimizers(self):
        super().reset_optimizers()
        self.optimizerDisc = get_optimizer(self.get_predictor("netBase"), lr=self.cfg_optim_discriminator.lr, weight_decay=self.cfg_optim_discriminator.weight_decay)
    
    def parse_dict_definition(self, dict_config, total_iter):
        '''
        The dict_config is a diction-based configuration with ascending order
        The key: value is the NUM_ITERATION_WEIGHT_BEGIN: WEIGHT
        For example,
        {0: 0.1, 1000: 0.2, 10000: 0.3}
        means at beginning, the weight is 0.1, from 1k iterations, weight is 0.2, and after 10k, weight is 0.3
        '''
        length = len(dict_config)
        all_iters = list(dict_config.keys())
        all_weights = list(dict_config.values())

        weight = all_weights[-1]

        for i in range(length-1):
            # this works for dict having at least two items, otherwise you don't need dict to set config
            iter_num = all_iters[i]
            iter_num_next = all_iters[i+1]
            if iter_num <= total_iter and total_iter < iter_num_next:
                weight = all_weights[i]
                break

        return weight
    
    def get_random_view_mask(self, w2c_pred, shape, prior_shape, num_frames, bins=360):
        delta_angle = 2 * np.pi / bins
        b = len(shape)

        # the render rotation matrix is different
        rand_degree = torch.randint(bins, [b])
        delta_angle = delta_angle * rand_degree
        delta_rot_matrix = []
        for i in range(b):
            angle = delta_angle[i].item()
            angle_matrix = torch.FloatTensor([
                [np.cos(angle),  0, np.sin(angle), 0],
                [0,              1, 0,             0],
                [-np.sin(angle), 0, np.cos(angle), 0],
                [0,              0, 0,             1],
            ]).to(self.accelerator.device)
            delta_rot_matrix.append(angle_matrix)
        delta_rot_matrix = torch.stack(delta_rot_matrix, dim=0)
        
        w2c = torch.FloatTensor(np.diag([1., 1., 1., 1]))
        w2c[:3, 3] = torch.FloatTensor([0, 0, -self.cfg_render.cam_pos_z_offset *1.4])
        w2c = w2c.repeat(b, 1, 1).to(self.accelerator.device)
        # use the predicted transition
        w2c_pred = w2c_pred.detach()
        w2c[:, :3, 3] = w2c_pred[:b][:, :3, 3]

        proj = util.perspective(self.cfg_render.fov / 180 * np.pi, 1, n=0.1, f=1000.0).repeat(b, 1, 1).to(self.accelerator.device)
        mvp = torch.bmm(proj, w2c)
        campos = -w2c[:, :3, 3]

        mvp = torch.matmul(mvp, delta_rot_matrix)
        campos = torch.matmul(delta_rot_matrix[:,:3,:3].transpose(2,1), campos[:,:,None])[:,:,0]

        resolution = (256, 256)
        mask_pred = self.render(
            render_modes=['shaded'],
            shape=shape, 
            texture=None,
            mvp=mvp, 
            w2c=w2c, 
            campos=campos, 
            resolution=resolution, 
            background='none', 
            im_features=None, 
            light=None, 
            prior_shape=prior_shape,
            dino_net=None,
            bsdf='diffuse', 
            two_sided_shading=False, 
            num_frames=num_frames,
            spp=None,
            class_vector=None,
        )   # a list
        mask_pred = mask_pred[0][:, 3:, :, :]
        mask_pred = mask_pred.clamp(0,1)
        

        out_aux = {
            'mask_random_pred': mask_pred,
            'rand_degree': rand_degree
        }

        return out_aux
    
    def compute_mask_disc_loss_gen(self, mask_gt, mask_pred, mask_random_pred, condition_feat=None):
        class_one_hot = condition_feat.detach()
        class_one_hot = class_one_hot.reshape(1, -1, 1, 1).repeat(mask_gt.shape[0], 1, mask_gt.shape[-2], mask_gt.shape[-1])

        # concat
        mask_gt = torch.cat([mask_gt, class_one_hot], dim=1)
        mask_pred = torch.cat([mask_pred, class_one_hot], dim=1)
        mask_random_pred = torch.cat([mask_random_pred, class_one_hot], dim=1)
        
        # mask shape are all [B,1,256,256]
        # the random view mask are False
        d_random_pred = self.netDisc(mask_random_pred)
        disc_loss = discriminator_architecture.bce_loss_target(d_random_pred, 1)  # in gen loss, train it to be real
        count = 1

        disc_loss_rv = disc_loss.detach()
        disc_loss_iv = 0.0
            
        if self.cfg_mask_discriminator.disc_iv:
            if self.cfg_mask_discriminator.disc_iv_label != 'Real': # consider the input view also fake
                d_iv = self.netDisc(mask_pred)
                disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 1) # so now we need to train them to be real
                disc_loss = disc_loss + disc_iv_loss
                count = count + 1
                disc_loss_iv = disc_iv_loss.detach()
        
        disc_loss = disc_loss / count

        # record the masks for discriminator training
        self.record_mask_gt = mask_gt.clone().detach()
        self.record_mask_iv = mask_pred.clone().detach()
        self.record_mask_rv = mask_random_pred.clone().detach()

        return {
            'mask_disc_loss': disc_loss,
            'mask_disc_loss_rv': disc_loss_rv,
            'mask_disc_loss_iv': disc_loss_iv,
        }
    
    def discriminator_step(self):
        mask_gt = self.record_mask_gt
        mask_pred = self.record_mask_iv
        mask_random_pred = self.record_mask_rv
        
        self.optimizerDisc.zero_grad()

        # the random view mask are False
        d_random_pred = self.netDisc(mask_random_pred)
        disc_loss = discriminator_architecture.bce_loss_target(d_random_pred, 0)  # in gen loss, train it to be real

        grad_loss = 0.0
        count = 1

        discriminator_loss_rv = disc_loss.detach()
        discriminator_loss_gt = 0.0
        discriminator_loss_iv = 0.
        d_gt = None
        d_iv = None
        
        if self.cfg_mask_discriminator.disc_gt:
            mask_gt.requires_grad_()
            d_gt = self.netDisc(mask_gt)
            if d_gt.requires_grad is False:
                # in the test case
                disc_gt_loss = discriminator_architecture.bce_loss_target(d_gt, 1)
            else:
                grad_penalty = self.cfg_mask_discriminator.disc_reg_mul * discriminator_architecture.compute_grad2(d_gt, mask_gt)
                disc_gt_loss = discriminator_architecture.bce_loss_target(d_gt, 1) + grad_penalty
                grad_loss += grad_penalty
            disc_loss = disc_loss + disc_gt_loss
            discriminator_loss_gt = disc_gt_loss
            count = count + 1
        
        if self.cfg_mask_discriminator.disc_iv:
            mask_pred.requires_grad_()
            d_iv = self.netDisc(mask_pred)
            if self.cfg_mask_discriminator.disc_iv_label == 'Real':
                if d_iv.requires_grad is False:
                    # in the test case
                    disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 1)
                else:    
                    grad_penalty = self.cfg_mask_discriminator.disc_reg_mul * discriminator_architecture.compute_grad2(d_iv, mask_pred)
                    disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 1) + grad_penalty
                    grad_loss += grad_penalty
                
            else:
                disc_iv_loss = discriminator_architecture.bce_loss_target(d_iv, 0)
            disc_loss = disc_loss + disc_iv_loss
            count = count + 1
            discriminator_loss_iv = disc_iv_loss
        
        disc_loss = disc_loss / count
        grad_loss = grad_loss / count

        self.discriminator_loss = disc_loss * self.cfg_mask_discriminator.discriminator_loss_weight
        self.discriminator_loss.backward()
        self.optimizerDisc.step()
        self.discriminator_loss = 0.
        return {
            'discriminator_loss': disc_loss,
            'discriminator_loss_rv': discriminator_loss_rv,
            'discriminator_loss_iv': discriminator_loss_iv,
            'discriminator_loss_gt': discriminator_loss_gt,
            'd_rv': d_random_pred,
            'd_iv': d_iv if d_iv is not None else None,
            'd_gt': d_gt if d_gt is not None else None,
        }, grad_loss
    
    def compute_regularizers(self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None, class_vector=None):
        losses = {}
        aux = {}
        losses.update(self.get_predictor("netBase").netShape.get_sdf_reg_loss(feats=class_vector))
        if arti_params is not None:
            losses['arti_reg_loss'] = (arti_params ** 2).mean()
        if deformation is not None:
            losses['deform_reg_loss'] = (deformation ** 2).mean()

        # Smooth losses
        if self.dataset.data_type == "sequence" and self.dataset.num_frames > 1:
            b, f = self.dataset.batch_size, self.dataset.num_frames
            if self.cfg_loss.deform_smooth_loss_weight > 0 and deformation is not None:
                losses["deform_smooth_loss"] = self.smooth_loss_fn(expandBF(deformation, b, f))
            if arti_params is not None:
                if self.cfg_loss.arti_smooth_loss_weight > 0:
                    losses["arti_smooth_loss"] = self.smooth_loss_fn(arti_params)
                if self.cfg_loss.artivel_smooth_loss_weight > 0:
                    artivel = arti_params[:, 1:, ...] - arti_params[:, :(f-1), ...]
                    losses["artivel_smooth_loss"] = self.smooth_loss_fn(artivel)
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
        if bbox.shape[2] == 9:
            # Fauna Dataset bbox
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, tmp_label = bbox.unbind(2)  # BxFx9
        elif bbox.shape[2] == 8:
            # in visualization using magicpony dataset for simplicity
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        else:
            raise NotImplementedError
    
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
        if self.get_predictor("netBase").netShape.grid_res != grid_res:
            self.get_predictor("netBase").netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training, batch=batch, bank_enc=self.get_predictor("netInstance").netEncoder)
        else:
            prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training, batch=batch, bank_enc=self.get_predictor("netInstance").netEncoder)
        
        class_vector = bank_embedding[0]

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training)
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(input_image, prior_shape, epoch, total_iter, is_training=is_training)  # first two dim dimensions already collapsed N=(B*F)
        # if not is_training and (batch_size != arti_params.shape[0] or num_frames != arti_params.shape[1]):
        #     # If b f sampled from vae different from training b f
        #     batch_size, num_frames = arti_params.shape[:2]
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
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    renders = self.render(
                        render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light, 
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames, 
                        class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                    )
            else:
                renders = self.render(
                    render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light, 
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames, 
                    class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                )
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
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
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
                        # loss_weight = loss_weight * self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier
                        loss_weight = self.parse_dict_definition(self.cfg_loss.dino_feat_im_loss_weight_dict, total_iter)
                        loss_weight = loss_weight * self.parse_dict_definition(self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier_dict, total_iter)
                    if name in ['mask_loss']:
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_mask_multiplier
                    if name in ['mask_inv_dt_loss']:
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_mask_inv_dt_multiplier
                    if loss_weight > 0:
                        logit_loss_target += loss * loss_weight

                    ## multiply the loss with probability of the rotation hypothesis (detached)
                    if self.get_predictor("netInstance").cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
                        loss_prob = rot_prob.detach().view(batch_size, num_frames)[:, :loss.shape[1]]  # handle edge case for flow loss with one frame less
                        loss = loss * loss_prob *self.get_predictor("netInstance").num_pose_hypos
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

        random_view_aux = None
        random_view_aux = self.get_random_view_mask(w2c, shape, prior_shape, num_frames)
        if (self.cfg_mask_discriminator.enable_iter[0] < total_iter) and (self.cfg_mask_discriminator.enable_iter[1] > total_iter):
            disc_loss = self.compute_mask_disc_loss_gen(
                mask_gt, mask_pred, random_view_aux['mask_random_pred'], condition_feat=class_vector
            )
            final_losses.update(disc_loss)

        ## regularizers
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                regularizers, aux = self.compute_regularizers(
                    arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                    posed_bones=forward_aux.get("posed_bones"),
                    class_vector=class_vector.detach() if class_vector is not None else None
                )
        else:
            regularizers, aux = self.compute_regularizers(
                arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                posed_bones=forward_aux.get("posed_bones"),
                class_vector=class_vector.detach() if class_vector is not None else None
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
            
            if name == 'dino_feat_im_loss':
                loss_weight = self.parse_dict_definition(self.cfg_loss.dino_feat_im_loss_weight_dict, total_iter)

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

    def log_visuals(self, log, logger):
        super().log_visuals(log, logger, sdf_feats=log.class_vector, text=self.all_category_names[int(log.tmp_label.reshape(-1)[0].item())])

        weights_for_emb = log.bank_embedding[2]['weights'] # [B, k]
        for i, weight_for_emb in enumerate(weights_for_emb.unbind(-1)):
            logger.add_histogram(log.logger_prefix+'bank_embedding/emb_weight_%d'%i, weight_for_emb, log.total_iter)
        
        indices_for_emb = log.bank_embedding[2]['pick_idx'] # [B, k]
        for i, idx_for_emb in enumerate(indices_for_emb.unbind(-1)):
            logger.add_histogram(log.logger_prefix+'bank_embedding/emb_idx_%d'%i, idx_for_emb, log.total_iter)
        return log