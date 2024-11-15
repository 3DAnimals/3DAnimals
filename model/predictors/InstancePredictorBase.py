from dataclasses import dataclass, field, asdict
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import networks
from model.utils import misc
from model.geometry.skinning import estimate_bones, skinning
from model.render import util, mesh, light


@dataclass
class ViTEncoderConfig:
    cout: int = 256
    which_vit: str = 'dino_vits8'
    pretrained: bool = False
    frozen: bool = False
    final_layer_type: str = 'conv'


@dataclass
class TextureConfig:
    texture_iter_range: List[int] = field(default_factory=lambda: [-1, -1])  # by default, turned on throughout
    cout: int = 9  # by default, only first three channels are used as albedo RGB
    num_layers: int = 5
    hidden_size: int = 64
    activation: str = "sigmoid"
    kd_minmax: List[List[float]] = field(default_factory=lambda: [[0., 0.], [0., 0.], [0., 0.]])
    ks_minmax: List[List[float]] = field(default_factory=lambda: [[0., 0.], [0., 0.], [0., 0.]])
    nrm_minmax: List[List[float]] = field(default_factory=lambda: [[-1., 1.], [-1., 1.], [0., 1.]])
    embed_concat_pts: bool = True
    embedder_freq: int = 10
    symmetrize: bool = False


@dataclass
class PoseConfig:
    architecture: str = 'encoder_dino_patch_key'
    cam_pos_z_offset: float = 10.
    fov: float = 25.
    max_trans_xy_range_ratio: float = 1.
    max_trans_z_range_ratio: float = 1.
    rot_rep: str = 'euler_angle'

    # if rot_rep == 'euler_angle':
    max_rot_x_range: float = 180.
    max_rot_y_range: float = 180.
    max_rot_z_range: float = 180.
    
    # if rot_rep == ['lookat', 'quadlookat', 'octlookat']:
    lookat_zeroy: bool = False
    
    # if rot_rep in ['quadlookat', 'octlookat']:
    rot_temp_scalar: float = 1.
    naive_probs_iter: int = 2000
    best_pose_start_iter: int = 6000


@dataclass
class DeformConfig:
    deform_iter_range: List[int] = field(default_factory=lambda: [-1, -1])
    num_layers: int = 5
    hidden_size: int = 64
    embed_concat_pts: bool = True
    embedder_freq: int = 10
    symmetrize: bool = False
    force_avg_deform: bool = True


@dataclass
class ArticulationConfig:
    articulation_iter_range: List[int] = field(default_factory=lambda: [-1, -1])
    architecture: str = 'mlp'
    num_layers: int = 4
    hidden_size: int = 64
    embedder_freq: int = 8
    bone_feature_mode: str = 'global'
    num_body_bones: int = 4  # assuming an even number of body bones
    body_bones_mode: str = 'z_minmax'
    num_legs: int = 0
    num_leg_bones: int = 0
    attach_legs_to_body_iter_range: List[int] = field(default_factory=lambda: [-1, -1])
    legs_to_body_joint_indices: Optional[List[int]] = None
    static_root_bones: bool = False
    skinning_temperature: float = 1.
    max_arti_angle: float = 60.
    constrain_legs: bool = False
    output_multiplier: float = 1.


@dataclass
class LightingConfig:
    num_layers: int = 5
    hidden_size: int = 64
    amb_diff_minmax: List[List[float]] = field(default_factory=lambda: [[0.0, 1.0], [0.5, 1.0]])


@dataclass
class InstancePredictorConfig:
    cfg_encoder: ViTEncoderConfig
    cfg_texture: TextureConfig
    cfg_pose: PoseConfig
    spatial_scale: float = 5.
    enable_deform: bool = False
    cfg_deform: Optional[DeformConfig] = None
    enable_articulation: bool = False
    cfg_articulation: Optional[ArticulationConfig] = None
    enable_lighting: bool = False
    cfg_light: Optional[LightingConfig] = None


class InstancePredictorBase(nn.Module):
    def __init__(self, cfg: InstancePredictorConfig):
        super().__init__()
        misc.load_cfg(self, cfg, InstancePredictorConfig)
        
        embedder_scalar = 2 * np.pi / self.spatial_scale * 0.9  # originally (-0.5*s, 0.5*s) rescale to (-pi, pi) * 0.9

        ## Image encoder
        self.netEncoder = networks.ViTEncoder(**asdict(self.cfg_encoder))
        vit_feat_dim = self.netEncoder.vit_feat_dim
        enc_feat_dim = self.cfg_encoder.cout
        
        ## Texture network
        kd_minmax = torch.FloatTensor(self.cfg_texture.kd_minmax)  # 3x2
        ks_minmax = torch.FloatTensor(self.cfg_texture.ks_minmax)  # 3x2
        nrm_minmax = torch.FloatTensor(self.cfg_texture.nrm_minmax)  # 3x2
        texture_min_max = torch.cat((kd_minmax, ks_minmax, nrm_minmax), dim=0)  # 9x2
        self.netTexture = networks.CoordMLP(
            3,  # x, y, z coordinates
            self.cfg_texture.cout,
            self.cfg_texture.num_layers,
            nf=self.cfg_texture.hidden_size,
            dropout=0,
            activation=self.cfg_texture.activation,
            min_max=texture_min_max,
            n_harmonic_functions=self.cfg_texture.embedder_freq,
            embedder_scalar=embedder_scalar,
            embed_concat_pts=self.cfg_texture.embed_concat_pts,
            extra_feat_dim=enc_feat_dim,
            symmetrize=self.cfg_texture.symmetrize
        )

        ## Pose network
        half_range = np.tan(self.cfg_pose.fov /2 /180 * np.pi) * self.cfg_pose.cam_pos_z_offset  # default=2.22
        self.max_trans_xyz_range = torch.FloatTensor([
            self.cfg_pose.max_trans_xy_range_ratio,
            self.cfg_pose.max_trans_xy_range_ratio,
            self.cfg_pose.max_trans_z_range_ratio]) * half_range
        if self.cfg_pose.rot_rep == 'euler_angle':
            pose_cout = 6  # 3 for rotation, 3 for translation
            self.max_rot_xyz_range = torch.FloatTensor([
                self.cfg_pose.max_rot_x_range,
                self.cfg_pose.max_rot_y_range,
                self.cfg_pose.max_rot_z_range]) /180 * np.pi
        elif self.cfg_pose.rot_rep == 'quaternion':
            pose_cout = 7  # 4 for rotation, 3 for translation
        elif self.cfg_pose.rot_rep == 'lookat':
            pose_cout = 6  # 3 for forward vector, 3 for translation
        elif self.cfg_pose.rot_rep == 'quadlookat':
            self.num_pose_hypos = 4
            pose_cout = (3 + 1) * self.num_pose_hypos + 3  # forward vector + prob logits for each hypothesis, 3 for translation
            self.orthant_signs = torch.FloatTensor([[1,1,1], [-1,1,1], [-1,1,-1], [1,1,-1]])  # 4x3
        elif self.cfg_pose.rot_rep == 'octlookat':
            self.num_pose_hypos = 8
            pose_cout = (3 + 1) * self.num_pose_hypos + 3  # forward vector + prob logits for each hypothesis, 3 for translation
            self.orthant_signs = torch.stack(torch.meshgrid([torch.arange(1, -2, -2)] *3), -1).view(-1, 3)  # 8x3
        else:
            raise NotImplementedError
        self.netPose = networks.Encoder32(cin=vit_feat_dim, cout=pose_cout, nf=256, activation=None)  # vit patches are 32x32
        
        ## Deformation network
        if self.enable_deform:
            self.netDeform = networks.CoordMLP(
                3,  # x, y, z coordinates
                3,  # dx, dy, dz deformation
                self.cfg_deform.num_layers,
                nf=self.cfg_deform.hidden_size,
                dropout=0,
                activation=None,
                min_max=None,
                n_harmonic_functions=self.cfg_deform.embedder_freq,
                embedder_scalar=embedder_scalar,
                embed_concat_pts=self.cfg_deform.embed_concat_pts,
                extra_feat_dim=enc_feat_dim,
                symmetrize=self.cfg_deform.symmetrize
            )

        ## Articulation network
        if self.enable_articulation:
            self.num_bones = self.cfg_articulation.num_body_bones + self.cfg_articulation.num_legs * self.cfg_articulation.num_leg_bones
            if self.cfg_articulation.bone_feature_mode == 'global':
                arti_feat_dim = enc_feat_dim
            elif self.cfg_articulation.bone_feature_mode == 'sample':
                arti_feat_dim = vit_feat_dim
            elif self.cfg_articulation.bone_feature_mode == 'sample+global':
                arti_feat_dim = vit_feat_dim + enc_feat_dim
            else:
                raise NotImplementedError
            self.netArticulation = networks.ArticulationNetwork(
                self.cfg_articulation.architecture,
                arti_feat_dim,
                posenc_dim=1 + 2 + 3 * 2,  # bone index + 2D mid bone position + 3D joint locations
                num_layers=self.cfg_articulation.num_layers,
                nf=self.cfg_articulation.hidden_size,
                n_harmonic_functions=self.cfg_articulation.embedder_freq,
                embedder_scalar=np.pi * 0.9  # originally (-1, 1) rescale to (-pi, pi) * 0.9
            )
            self.kinematic_tree_epoch = -1  # initialize to -1 to force compute kinematic tree at first epoch
        
        ## Lighting network
        if self.enable_lighting:
            self.netLight = light.DirectionalLight(
                enc_feat_dim,
                self.cfg_light.num_layers,
                self.cfg_light.hidden_size,
                intensity_min_max=torch.FloatTensor(self.cfg_light.amb_diff_minmax)
            )

    def forward_encoder(self, images):
        images_in = images.view(-1, *images.shape[2:]) * 2 - 1  # (B*F)xCxHxW rescale to (-1, 1)
        feat_out, feat_key, patch_out, patch_key = self.netEncoder(images_in, return_patches=True)
        return feat_out, feat_key, patch_out, patch_key
    
    def forward_pose(self, patch_out, patch_key):
        if self.cfg_pose.architecture == 'encoder_dino_patch_key':
            pose = self.netPose(patch_key)  # Shape: (B, latent_dim)
        elif self.cfg_pose.architecture == 'encoder_dino_patch_out':
            pose = self.netPose(patch_out)  # Shape: (B, latent_dim)
        else:
            raise NotImplementedError
        
        ## xyz translation
        trans_pred = pose[..., -3:].tanh() * self.max_trans_xyz_range.to(pose.device)

        ## rotation
        if self.cfg_pose.rot_rep == 'euler_angle':
            rot_pred = pose[..., :3].tanh() * self.max_rot_xyz_range.to(pose.device)

        elif self.cfg_pose.rot_rep == 'quaternion':
            quat_init = torch.FloatTensor([0.01,0,0,0]).to(pose.device)
            rot_pred = pose[..., :4] + quat_init
            rot_pred = nn.functional.normalize(rot_pred, p=2, dim=-1)
            # rot_pred = torch.cat([rot_pred[...,:1].abs(), rot_pred[...,1:]], -1)  # make real part non-negative
            rot_pred = rot_pred * rot_pred[...,:1].sign()  # make real part non-negative

        elif self.cfg_pose.rot_rep == 'lookat':
            vec_forward = pose[..., :3]
            if self.cfg_pose.lookat_zeroy:
                vec_forward = vec_forward * torch.FloatTensor([1,0,1]).to(pose.device)
            vec_forward = nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = vec_forward

        elif self.cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
            rots_pred = pose[..., :self.num_pose_hypos*4].view(-1, self.num_pose_hypos, 4)  # (B*F, K, 4)
            rots_logits = rots_pred[..., :1]
            vec_forward = rots_pred[..., 1:4]

            def softplus_with_init(x, init=0.5):
                assert np.abs(init) > 1e-8, "initial value should be non-zero"
                beta = np.log(2) / init
                return nn.functional.softplus(x, beta=beta)
            
            xs, ys, zs = vec_forward.unbind(-1)
            xs = softplus_with_init(xs, init=0.5)  # initialize to 0.5
            if self.cfg_pose.rot_rep == 'octlookat':
                ys = softplus_with_init(ys, init=0.5)  # initialize to 0.5
            if self.cfg_pose.lookat_zeroy:
                ys = ys * 0
            zs = softplus_with_init(zs, init=0.5)  # initialize to 0.5
            vec_forward = torch.stack([xs, ys, zs], -1)
            vec_forward = vec_forward * self.orthant_signs.to(pose.device)
            vec_forward = nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = torch.cat([rots_logits, vec_forward], -1).view(-1, self.num_pose_hypos*4)  # (B*F, K*4)

        else:
            raise NotImplementedError
        
        pose = torch.cat([rot_pred, trans_pred], -1)
        return pose
    
    def forward_deformation(self, shape, feat=None, **kwargs):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat), 1, 1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3), multiply by 0.1 to minimize disruption when initially enabled
        shape = shape.deform(deformation)
        return shape, deformation

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
                legs_to_body_joint_indices=self.cfg_articulation.legs_to_body_joint_indices
            )
            self.kinematic_tree_epoch = epoch
        else:
            bones = estimate_bones(
                verts.detach(), n_body_bones=self.cfg_articulation.num_body_bones,
                n_legs=self.cfg_articulation.num_legs, n_leg_bones=self.cfg_articulation.num_leg_bones,
                body_bones_mode=self.cfg_articulation.body_bones_mode, compute_kinematic_chain=False, aux=self.bone_aux
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
                patch_feat, bones_mid_pos_2d.view(batch_size * num_frames, 1, -1, 2), mode='bilinear', align_corners=False
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

    def apply_articulation_constraints(self, articulation_angles):
        articulation_angles *= self.cfg_articulation.output_multiplier
        if self.cfg_articulation.static_root_bones:
            root_bones = [self.cfg_articulation.num_body_bones // 2 - 1, self.cfg_articulation.num_body_bones - 1]
            tmp_mask = torch.ones_like(articulation_angles)
            tmp_mask[:, :, root_bones] = 0
            articulation_angles = articulation_angles * tmp_mask
        articulation_angles = articulation_angles.tanh()
        if self.cfg_articulation.constrain_legs:
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
        articulation_angles = articulation_angles * self.cfg_articulation.max_arti_angle / 180 * np.pi
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
        articulation_angles = self.apply_articulation_constraints(articulation_angles)

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
    
    def get_camera_extrinsics_from_pose(self, pose, znear=0.1, zfar=1000., offset_extra=None):
        N = len(pose)
        pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)  # to be compatible with pytorch3d
        if offset_extra is not None:
            cam_pos_offset = torch.FloatTensor([0, 0, -self.cfg_pose.cam_pos_z_offset - offset_extra]).to(pose.device)
        else:
            cam_pos_offset = torch.FloatTensor([0, 0, -self.cfg_pose.cam_pos_z_offset]).to(pose.device)
        pose_T = pose[:, -3:] + cam_pos_offset[None, None, :]
        pose_T = pose_T.view(N, 3, 1)
        pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
        w2c = torch.cat([pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)], axis=1)  # Nx4x4
        proj = util.perspective(self.cfg_pose.fov / 180 * np.pi, 1, znear, zfar)[None].to(pose.device)  # assuming square images
        mvp = torch.matmul(proj, w2c)
        campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
        return mvp, w2c, campos

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
        temp = 1 / np.clip(total_iter / 1000 / rot_temp_scalar, 1., 100.)

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

    def forward(self, images=None, prior_shape=None, epoch=None, total_iter=None, is_training=True, **kwargs):
        batch_size, num_frames = images.shape[:2]
        feat_out, feat_key, patch_out, patch_key = self.forward_encoder(images)  # first two dimensions are collapsed N=(B*F)
        shape = prior_shape
        texture = self.netTexture

        poses_raw = self.forward_pose(patch_out, patch_key)
        assert self.cfg_pose.rot_rep in ['quadlookat', 'octlookat'], "modify the forward process to support other rotation representations"
        pose_raw, pose, multi_hypothesis_aux = self.sample_pose_hypothesis_from_quad_predictions(
            poses_raw, total_iter, rot_temp_scalar=self.cfg_pose.rot_temp_scalar, num_hypos=self.num_pose_hypos,
            naive_probs_iter=self.cfg_pose.naive_probs_iter, best_pose_start_iter=self.cfg_pose.best_pose_start_iter,
            random_sample=(is_training and not (hasattr(self, "cfg_motion_vae") and self.enable_motion_vae))
        )
        mvp, w2c, campos = self.get_camera_extrinsics_from_pose(pose)

        deformation = None
        if self.enable_deform and in_range(total_iter, self.cfg_deform.deform_iter_range):
            shape, deformation = self.forward_deformation(shape, feat_key, batch_size=batch_size, num_frames=num_frames)
        else:  # Dummy operations for accelerator ddp
            shape.v_pos += sum([p.sum() * 0 for p in self.netDeform.parameters()])
            shape.v_pos += sum([p.sum() * 0 for p in self.netEncoder.parameters()])
        arti_params, articulation_aux = None, {}
        if self.enable_articulation and in_range(total_iter, self.cfg_articulation.articulation_iter_range):
            for p in self.netArticulation.parameters():
                p.requires_grad = True
            shape, arti_params, articulation_aux = self.forward_articulation(shape, feat_key, patch_key, mvp, w2c, batch_size, num_frames, epoch, total_iter)
        else:  # Dummy operations for accelerator ddp
            shape.v_pos += sum([p.sum() * 0 for p in self.netArticulation.parameters()])

        light = self.netLight if self.enable_lighting else None

        aux = multi_hypothesis_aux
        aux.update(articulation_aux)

        return shape, pose_raw, pose, mvp, w2c, campos, texture, feat_out, deformation, arti_params, light, aux


## utility functions
def in_range(x, range):
    return misc.in_range(x, range, default_indicator=-1)


def lookat_forward_to_rot_matrix(vec_forward, up=[0,1,0]):
    # vec_forward = nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward  -- assumed normalized
    up = torch.FloatTensor(up).to(vec_forward.device)
    vec_right = up.expand_as(vec_forward).cross(vec_forward, dim=-1)
    vec_right = nn.functional.normalize(vec_right, p=2, dim=-1)
    vec_up = vec_forward.cross(vec_right, dim=-1)
    vec_up = nn.functional.normalize(vec_up, p=2, dim=-1)
    rot_mat = torch.stack([vec_right, vec_up, vec_forward], -2)
    return rot_mat
