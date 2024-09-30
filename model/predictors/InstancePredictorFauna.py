from dataclasses import dataclass, field, asdict
from typing import List, Optional
from copy import deepcopy
import numpy as np
import torch
from einops import rearrange
import torch.nn.functional as F
from model.utils import misc
from model.geometry.skinning import skinning
from model.render import mesh
from ..predictors import InstancePredictorConfig, InstancePredictorBase, in_range, estimate_bones, lookat_forward_to_rot_matrix


@dataclass
class FaunaInstanceAdditionalConfig:
    iter_leg_rotation_start: int = 300000
    forbid_leg_rotate: bool = True
    small_leg_angle: int = True
    reg_body_rotate_mult: float = 0.1
    bone_y_threshold: float = 0.4
    nozeroy_start: int = 20000


@dataclass
class FaunaInstancePredictorConfig(InstancePredictorConfig):
    cfg_additional: FaunaInstanceAdditionalConfig = FaunaInstanceAdditionalConfig()


class InstancePredictorFauna(InstancePredictorBase):
    def __init__(self, cfg: FaunaInstancePredictorConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, FaunaInstancePredictorConfig)
        self.netTexture.in_layer_relu = True
        self.netArticulation.enable_articulation_idadd = True

    @staticmethod
    def sample_pose_hypothesis_from_quad_predictions(
        poses_raw, total_iter, rot_temp_scalar=1., num_hypos=4, naive_probs_iter=2000, best_pose_start_iter=6000,
        random_sample=True
    ):
        rots_pred = poses_raw[..., :num_hypos * 4].view(-1, num_hypos, 4)  # NxKx4
        N = len(rots_pred)
        rots_logits = rots_pred[..., 0]  # Nx4
        rots_pred = rots_pred[..., 1:4]
        trans_pred = poses_raw[..., -3:]
        temp = 1 / np.clip(total_iter / 1000 / rot_temp_scalar, 1, 10.)

        rots_probs = torch.nn.functional.softmax(-rots_logits / temp, dim=1)  # NxK
        naive_probs = torch.ones(num_hypos).to(rots_logits.device)
        naive_probs = naive_probs / naive_probs.sum()
        naive_probs_weight = np.clip(1 - (total_iter - naive_probs_iter) / 2000, 0, 1)
        rots_probs = naive_probs.view(1, num_hypos) * naive_probs_weight + rots_probs * (1 - naive_probs_weight)
        best_rot_idx = torch.argmax(rots_probs, dim=1)  # N

        if random_sample:
            rand_rot_idx = torch.randperm(N, device=poses_raw.device) % num_hypos  # N
            best_flag = (torch.randperm(N, device=poses_raw.device) / N < np.clip((total_iter - best_pose_start_iter) / 2000, 0, 0.8)).long()
            rand_flag = 1 - best_flag
            rot_idx = best_rot_idx * best_flag + rand_rot_idx * (1 - best_flag)
        else:
            rand_flag = torch.zeros_like(best_rot_idx)
            rot_idx = best_rot_idx
        rot_pred = torch.gather(rots_pred, 1, rot_idx[:, None, None].expand(-1, 1, 3))[:, 0]  # Nx3
        pose_raw = torch.cat([rot_pred, trans_pred], -1)
        rot_prob = torch.gather(rots_probs, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N
        rot_logit = torch.gather(rots_logits, 1, rot_idx[:, None].expand(-1, 1))[:, 0]  # N

        rot_mat = lookat_forward_to_rot_matrix(rot_pred, up=[0, 1, 0])
        pose = torch.cat([rot_mat.view(N, -1), pose_raw[:, 3:]], -1)  # flattened to Nx12
        pose_aux = {
            'rot_idx': rot_idx,
            'rot_prob': rot_prob,
            'rot_logit': rot_logit,
            'rots_probs': rots_probs,
            'rand_pose_flag': rand_flag
        }
        return pose_raw, pose, pose_aux

    def get_bones(self, verts, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter):
        """Get estimated bones from prior shape and corresponding bone feature and encoding"""
        ## recompute kinematic tree at the beginning of each epoch
        if self.kinematic_tree_epoch != epoch:
            attach_legs_to_body = in_range(total_iter, self.cfg_articulation.attach_legs_to_body_iter_range)
            bones, self.kinematic_tree, self.bone_aux = estimate_bones(
                verts.detach(), n_body_bones=self.cfg_articulation.num_body_bones,
                n_legs=self.cfg_articulation.num_legs, n_leg_bones=self.cfg_articulation.num_leg_bones,
                body_bones_mode=self.cfg_articulation.body_bones_mode, compute_kinematic_chain=True,
                attach_legs_to_body=attach_legs_to_body,
                legs_to_body_joint_indices=self.cfg_articulation.legs_to_body_joint_indices,
                bone_y_threshold=self.cfg_additional.bone_y_threshold
            )
            # estimate bones every iteration for fauna
            # self.kinematic_tree_epoch = epoch
        else:
            bones = estimate_bones(
                verts.detach(), n_body_bones=self.cfg_articulation.num_body_bones,
                n_legs=self.cfg_articulation.num_legs, n_leg_bones=self.cfg_articulation.num_leg_bones,
                body_bones_mode=self.cfg_articulation.body_bones_mode, compute_kinematic_chain=False, aux=self.bone_aux,
                bone_y_threshold=self.cfg_additional.bone_y_threshold
            )

        ## 2D location of bone mid point
        bones_pos = bones  # Shape: (B, F, K, 2, 3)
        if batch_size > bones_pos.shape[0] or num_frames > bones_pos.shape[1]:
            assert bones_pos.shape[0] == 1 and bones_pos.shape[1] == 1, \
                "canonical mesh should have batch_size=1 and num_frames=1"
            bones_pos = bones_pos.repeat(batch_size, num_frames, 1, 1, 1)
        num_bones = bones_pos.shape[2]
        bones_pos = bones_pos.view(batch_size * num_frames, num_bones, 2, 3)  # NxKx2x3
        bones_mid_pos = bones_pos.mean(2)  # NxKx3
        bones_mid_pos_world4 = torch.cat([bones_mid_pos, torch.ones_like(bones_mid_pos[..., :1])], -1)  # NxKx4
        bones_mid_pos_clip4 = bones_mid_pos_world4 @ mvp.transpose(-1, -2)
        bones_mid_pos_2d = bones_mid_pos_clip4[..., :2] / bones_mid_pos_clip4[..., 3:4]
        bones_mid_pos_2d = bones_mid_pos_2d.detach()  # we don't want gradient to flow through the camera projection

        ## 3D locations of two bone end points in camera space
        bones_pos_world4 = torch.cat([bones_pos, torch.ones_like(bones_pos[..., :1])], -1)  # NxKx2x4
        bones_pos_cam4 = bones_pos_world4 @ w2c[:, None].transpose(-1, -2)
        bones_pos_cam3 = bones_pos_cam4[..., :3] / bones_pos_cam4[..., 3:4]
        bones_pos_cam3 = bones_pos_cam3 + torch.FloatTensor(
            [0, 0, self.cfg_pose.cam_pos_z_offset]
        ).to(bones_pos_cam3.device).view(1, 1, 1, 3)
        # (-1, 1), NxKx(2*3)
        bones_pos_3d = bones_pos_cam3.view(batch_size * num_frames, num_bones, 2 * 3) / self.spatial_scale * 2

        ## bone index
        bones_idx = torch.arange(num_bones).to(bones_pos.device)
        bones_idx_in = ((bones_idx[None, :, None] + 0.5) / num_bones * 2 - 1).repeat(batch_size * num_frames, 1, 1)
        bones_pos_in = torch.cat([bones_mid_pos_2d, bones_pos_3d, bones_idx_in], -1)
        bones_pos_in = bones_pos_in.detach()  # we don't want gradient to flow through the camera pose

        if feat is not None and patch_feat is not None:
            global_feat = feat[:, None].repeat(1, num_bones, 1)  # (BxF, K, feat_dim)
            local_feat = F.grid_sample(
                patch_feat, bones_mid_pos_2d.view(batch_size * num_frames, 1, -1, 2), mode='bilinear'
            ).squeeze(dim=-2).permute(0, 2, 1)  # (BxF, K, feat_dim)
            if self.cfg_articulation.bone_feature_mode == "global":
                bones_feat = global_feat
            elif self.cfg_articulation.bone_feature_mode == "sample":
                bones_feat = local_feat
            elif self.cfg_articulation.bone_feature_mode == "sample+global":
                bones_feat = torch.cat([global_feat, local_feat], -1)
            else:
                raise NotImplementedError
        else:
            bones_feat = None
        return bones, bones_feat, bones_pos_in
    
    def apply_fauna_articulation_regularizer(self, articulation_angles, total_iter):
        # new regularizations, for bottom 2 bones of each leg, they can only rotate around x-axis, 
        # and for the toppest bone of legs, restrict its angles in a smaller range
        if (self.cfg_additional.iter_leg_rotation_start > 0) and (total_iter > self.cfg_additional.iter_leg_rotation_start):
            if self.cfg_additional.forbid_leg_rotate:
                if self.cfg_additional.small_leg_angle:
                    # regularize the rotation angle of first leg bones
                    leg_bones_top = [8, 11, 14, 17]
                    # leg_bones_top = [10, 13, 16, 19]
                    tmp_mask = torch.zeros_like(articulation_angles)
                    tmp_mask[:, :, leg_bones_top, 1] = 1
                    tmp_mask[:, :, leg_bones_top, 2] = 1
                    articulation_angles = tmp_mask * (articulation_angles * 0.05) + (1 - tmp_mask) * articulation_angles

                leg_bones_bottom = [9, 10, 12, 13, 15, 16, 18, 19]
                # leg_bones_bottom = [8, 9, 11, 12, 14, 15, 17, 18]
                tmp_mask = torch.ones_like(articulation_angles)
                tmp_mask[:, :, leg_bones_bottom, 1] = 0
                tmp_mask[:, :, leg_bones_bottom, 2] = 0
                # tmp_mask[:, :, leg_bones_bottom, 0] = 0.3
                articulation_angles = tmp_mask * articulation_angles
        
        # this after self.constrain_legs=True or False
        articulation_angles = articulation_angles * self.cfg_articulation.max_arti_angle / 180 * np.pi
        
        # check if regularize the leg-connecting body bones z-rotation first
        # then check if regularize all the body bones z-rotation
        # regularize z-rotation using 0.1 in pi-space
        body_rotate_mult = self.cfg_additional.reg_body_rotate_mult
        body_rotate_mult = body_rotate_mult * 180 * 1.0 / (self.cfg_articulation.max_arti_angle * np.pi)     # the max angle = mult*original_max_angle
        
        body_bones_mask = [0, 1, 2, 3, 4, 5, 6, 7]
        tmp_body_mask = torch.zeros_like(articulation_angles)
        tmp_body_mask[:, :, body_bones_mask, 2] = 1
        articulation_angles = tmp_body_mask * (articulation_angles * body_rotate_mult) + (1 - tmp_body_mask) * articulation_angles

        return articulation_angles
    
    def apply_articulation_constraints(self, articulation_angles, total_iter):
        if self.cfg_articulation.static_root_bones:
            root_bones = [self.cfg_articulation.num_body_bones // 2 - 1, self.cfg_articulation.num_body_bones - 1]
            tmp_mask = torch.ones_like(articulation_angles)
            tmp_mask[:, :, root_bones] = 0
            articulation_angles = articulation_angles * tmp_mask
        
        if total_iter <= self.cfg_additional.iter_leg_rotation_start:
            self.constrain_legs = True
        else:
            self.constrain_legs = False

        # if self.cfg_articulation.constrain_legs:
        if self.constrain_legs:
            leg_bones_idx = self.cfg_articulation.num_body_bones \
                + np.arange(self.cfg_articulation.num_leg_bones * self.cfg_articulation.num_legs)
            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_idx, 2] = 1  # twist / rotation around z axis
            articulation_angles = tmp_mask * (articulation_angles * 0.3) \
                + (1 - tmp_mask) * articulation_angles  # limit to (-0.3, 0.3)

            tmp_mask = torch.zeros_like(articulation_angles)
            tmp_mask[:, :, leg_bones_idx, 1] = 1  # side bending / rotation around y axis
            articulation_angles = tmp_mask * (articulation_angles * 0.3) \
                + (1 - tmp_mask) * articulation_angles  # limit to (-0.3, 0.3)
        
        return articulation_angles
    
    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter):
        verts = shape.v_pos
        if len(verts) == batch_size * num_frames:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])  # BxFxNx3
        else:
            verts = verts[None]  # 1x1xNx3

        bones, bones_feat, bones_pos_in = self.get_bones(
            verts, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter
        )

        articulation_angles = self.netArticulation(
            bones_feat, bones_pos_in
        ).view(batch_size, num_frames, bones.shape[2], 3)  # (B, F, K, 3)
        articulation_angles *= self.cfg_articulation.output_multiplier
        articulation_angles = articulation_angles.tanh()

        articulation_angles = self.apply_articulation_constraints(articulation_angles, total_iter)

        articulation_angles = self.apply_fauna_articulation_regularizer(articulation_angles, total_iter)

        verts_articulated, aux = skinning(
            verts, bones, self.kinematic_tree, articulation_angles, output_posed_bones=True,
            temperature=self.cfg_articulation.skinning_temperature
        )
        verts_articulated = verts_articulated.view(batch_size*num_frames, *verts_articulated.shape[2:])  # (B*F)xNx3
        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated):
            v_tex = v_tex.repeat(len(verts_articulated), 1, 1)
        articulated_shape = mesh.make_mesh(
            verts_articulated, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material
        )
        return articulated_shape, articulation_angles, aux
    
    def forward(self, images=None, prior_shape=None, epoch=None, total_iter=None, is_training=True):
        if total_iter >= self.cfg_additional.nozeroy_start:
            self.cfg_pose.lookat_zeroy = False
        shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, deformation, arti_params, light, aux = super().forward(images, prior_shape, epoch, total_iter, is_training)
        return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, deformation, arti_params, light, aux
