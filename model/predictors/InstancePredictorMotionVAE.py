from dataclasses import dataclass
import torch
import random
import numpy as np
from einops import rearrange, repeat
from model.utils import misc
from model.geometry.skinning import skinning
from model.render import mesh
from ..networks import ArticulationVAE
from ..predictors import InstancePredictorConfig, InstancePredictorBase


@dataclass
class MotionVAEConfig:
    latent_dim: int = 256
    z_token_num: int = 1
    transformer_layer_num: int = 4
    pe_dropout: float = 0.


@dataclass
class InstancePredictorMotionVAEConfig(InstancePredictorConfig):
    enable_motion_vae: bool = True
    cfg_motion_vae: MotionVAEConfig = MotionVAEConfig()
    render_gt_mesh: bool = False


class InstancePredictorMotionVAE(InstancePredictorBase):
    def __init__(self, cfg: InstancePredictorMotionVAEConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, InstancePredictorMotionVAEConfig)
        if self.enable_motion_vae:
            self.netVAE = ArticulationVAE(
                njoints=20,
                feat_dim=640,
                pos_dim=1 + 2 + 3 * 2,
                n_harmonic_functions=8,
                harmonic_omega0=np.pi * 0.9,
                latent_dim=self.cfg_motion_vae.latent_dim,
                pe_dropout=self.cfg_motion_vae.pe_dropout,
                transformer_layer_num=self.cfg_motion_vae.transformer_layer_num,
                z_token_num=self.cfg_motion_vae.z_token_num
            )

    def forward_deformation(self, shape, feat=None, batch_size=None, num_frames=None):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat), 1, 1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3), multiply by 0.1 to minimize disruption when initially enabled
        if deformation.shape[0] > 1 and self.cfg_deform.force_avg_deform:
            assert batch_size is not None and num_frames is not None
            assert deformation.shape[0] == batch_size * num_frames
            deformation = deformation.view(batch_size, num_frames, *deformation.shape[1:])
            deformation = deformation.mean(dim=1, keepdim=True)
            deformation = deformation.repeat(1,num_frames,*[1]*(deformation.dim()-2))
            deformation = deformation.view(batch_size*num_frames, *deformation.shape[2:])
        shape = shape.deform(deformation)
        return shape, deformation

    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter, **kwargs):
        """
        Forward propagation of articulation. For each bone, the network takes:
        1) the 3D location of the bone;
        2) the feature of the patch which the bone is projected to; and 3) an encoding of the bone's index to predict
            the bone's rotation (represented by an Euler angle).
        Args:
            shape: a Mesh object, whose v_pos has batch size BxF or 1.
            feat: the feature of the patches. Shape: (BxF, feat_dim, num_patches_per_axis, num_patches_per_axis)
            mvp: the model-view-projection matrix. Shape: (BxF, 4, 4)
        Returns:
            shape: a Mesh object, whose v_pos has batch size BxF (collapsed).
            articulation_angles: the predicted bone rotations. Shape: (B, F, num_bones, 3)
            aux: a dictionary containing auxiliary information.
        """
        if not self.enable_motion_vae:
            return super().forward_articulation(
                shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter
            )
        verts = shape.v_pos
        if len(verts) == batch_size * num_frames:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])  # BxFxNx3
        else:
            verts = verts[None]  # 1x1xNx3

        bones, bones_feat, bones_pos_in = self.get_bones(
            verts, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter
        )

        # Get gt articulation angles
        with torch.no_grad():
            articulation_angles_gt = self.netArticulation(
                bones_feat, bones_pos_in
            ).view(batch_size, num_frames, bones.shape[2], 3)
            articulation_angles_gt = self.apply_articulation_constraints(articulation_angles_gt)
            self.articulation_angles_gt = articulation_angles_gt.detach()

        # forward vae using bones feat get pred articulation angles
        articulation_angles_pred, input_vae, self.mu_vae, self.log_var_vae = self.netVAE(
            bones_feat, bones_pos_in, num_frames, batch_size
        )
        articulation_angles_pred = self.apply_articulation_constraints(articulation_angles_pred)
        self.articulation_angles_pred = articulation_angles_pred

        # skinning and make pred shape
        verts_articulated_pred, aux = skinning(
            verts, bones, self.kinematic_tree, articulation_angles_pred, output_posed_bones=True,
            temperature=self.cfg_articulation.skinning_temperature
        )
        verts_articulated_pred = verts_articulated_pred.view(batch_size * num_frames, *verts_articulated_pred.shape[2:])
        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated_pred):
            v_tex = v_tex.repeat(len(verts_articulated_pred), 1, 1)
        articulated_shape_pred = mesh.make_mesh(
            verts_articulated_pred, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material
        )

        # skinning and make gt shape
        if self.render_gt_mesh:
            verts_articulated_gt, aux_gt = skinning(
                verts, bones, self.kinematic_tree, articulation_angles_gt, output_posed_bones=True,
                temperature=self.cfg_articulation.skinning_temperature
            )
            verts_articulated_gt = verts_articulated_gt.view(batch_size * num_frames, *verts_articulated_gt.shape[2:])
            v_tex = shape.v_tex
            if len(v_tex) != len(verts_articulated_gt):
                v_tex = v_tex.repeat(len(verts_articulated_gt), 1, 1)
            articulated_shape_gt = mesh.make_mesh(
                verts_articulated_gt, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material
            )
            self.articulated_shape_gt = articulated_shape_gt
            for k, v in aux_gt.items():
                aux[f"{k}_gt"] = v
        return articulated_shape_pred, articulation_angles_pred, aux

    def generate_articulation(self, shape, mvp, w2c, num_sequence, num_frames, epoch, total_iter):
        verts = shape.v_pos
        if len(verts) == num_sequence * num_frames:
            verts = verts.view(num_sequence, num_frames, *verts.shape[1:])  # BxFxNx3
        else:
            verts = verts[None]  # 1x1xNx3

        bones, _, bones_pos_in = self.get_bones(
            verts, None, None, mvp, w2c, num_sequence, num_frames, epoch, total_iter
        )

        articulation_angles = self.netVAE.sample(num_sequence, num_frames)
        articulation_angles = self.apply_articulation_constraints(articulation_angles)

        verts = repeat(verts, "1 1 v d -> b f v d", b=num_sequence, f=num_frames)
        bones = repeat(bones, "1 1 k p d -> b f k p d", b=num_sequence, f=num_frames)
        verts_articulated, aux = skinning(
            verts, bones, self.kinematic_tree, articulation_angles, output_posed_bones=True,
            temperature=self.cfg_articulation.skinning_temperature
        )
        verts_articulated = rearrange(verts_articulated, "b f v d -> (b f) v d")

        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated):
            v_tex = v_tex.repeat(len(verts_articulated), 1, 1)
        shape = mesh.make_mesh(
            verts_articulated, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material
        )
        return shape, articulation_angles, aux

    def generate(
        self, images=None, prior_shape=None, epoch=None, total_iter=None, num_sequence=1, num_frames=10,
        **kwargs
    ):
        images = rearrange(images, "b f c h w -> (b f) c h w")
        idx = random.randrange(len(images))  # Use one frame to add generated articulation
        images = images[idx, None, None]
        feat_out, feat_key, patch_out, patch_key = self.forward_encoder(images)
        shape = prior_shape
        texture = self.netTexture

        poses_raw = self.forward_pose(patch_out, patch_key)
        assert self.cfg_pose.rot_rep in ["quadlookat", "octlookat"], \
            "modify the forward process to support other rotation representations"
        pose_raw, pose, multi_hypothesis_aux = self.sample_pose_hypothesis_from_quad_predictions(
            poses_raw, total_iter=float("inf"), rot_temp_scalar=self.cfg_pose.rot_temp_scalar, num_hypos=self.num_pose_hypos,
            random_sample=False
        )
        mvp, w2c, campos = self.get_camera_extrinsics_from_pose(pose)
        deformation = None
        if self.enable_deform:
            shape, deformation = self.forward_deformation(
                shape, feat_key, batch_size=num_sequence, num_frames=num_frames
            )
        shape, arti_params, articulation_aux = self.generate_articulation(
            shape, mvp, w2c, num_sequence, num_frames, epoch, total_iter
        )

        light = self.netLight if self.enable_lighting else None
        aux = multi_hypothesis_aux
        aux.update(articulation_aux)

        repeat_bf = lambda x: repeat(x, "1 ... -> b ...", b=num_sequence*num_frames) if x is not None else None
        pose_raw, pose, mvp, w2c, campos, feat_out, deformation = map(
            repeat_bf, [pose_raw, pose, mvp, w2c, campos, feat_out, deformation]
        )
        aux["rot_idx"], aux["rot_prob"], aux["rot_logit"], aux["rots_probs"], aux["rand_pose_flag"] = map(
            repeat_bf, [aux["rot_idx"], aux["rot_prob"], aux["rot_logit"], aux["rots_probs"], aux["rand_pose_flag"]]
        )

        return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, deformation, arti_params, light, aux
