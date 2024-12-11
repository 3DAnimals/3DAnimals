import hydra
import sys
import shutil
import os
import os.path as osp
sys.path.append(osp.abspath(osp.join(__file__, "../../")))

from pathlib import Path
from omegaconf import OmegaConf
import glob
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from moviepy.editor import ImageSequenceClip
import numpy as np
from pytorch3d import transforms
import torch
import torchvision
import nvdiffrast.torch as dr
from model import build_model
from model.utils.misc import setup_runtime
from model.render.mesh import make_mesh
import model.render.renderutils as ru
from model.geometry.skinning import estimate_bones, skinning, euler_angles_to_matrix
from model.render import util
from model.dataloaders import get_data_loaders


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def save_txts(arrays, base_names, out_dir, suffix=""):
    for arr, base_name in zip(arrays, base_names):
        arr = arr.cpu().numpy()
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(osp.join(out_dir, base_name + suffix + ".txt"), arr)


def save_images(images, mask_pred, base_names, out_dir, suffix="", mode="transparent"):
    if mask_pred is None:
        mask_pred = [None] * len(images)
    for img, mask, base_name in zip(images, mask_pred, base_names):
        img = img.clamp(0, 1)
        if mask is not None:
            mask = mask.clamp(0, 1)
            if mode == "white":
                img = img * mask + 1 * (1 - mask)
            else:
                img = torch.cat([img, mask[0:1]], 0)
        
        img = img.permute(1, 2, 0).cpu().numpy()
        img = Image.fromarray(np.uint8(img * 255))
        os.makedirs(out_dir, exist_ok=True)
        img.save(osp.join(out_dir, base_name + suffix + ".png"))


class FixedDirectionLight(torch.nn.Module):
    def __init__(self, direction, amb, diff):
        super(FixedDirectionLight, self).__init__()
        self.light_dir = direction
        self.amb = amb
        self.diff = diff
        self.is_hacking = not (isinstance(self.amb, float)
                               or isinstance(self.amb, int))

    def forward(self, feat):
        batch_size = feat.shape[0]
        if self.is_hacking:
            return torch.concat([self.light_dir, self.amb, self.diff], -1)
        else:
            return torch.concat([self.light_dir, torch.FloatTensor([self.amb, self.diff]).to(self.light_dir.device)], -1).expand(batch_size, -1)

    def shade(self, feat, kd, normal):
        light_params = self.forward(feat)
        light_dir = light_params[..., :3][:, None, None, :]
        int_amb = light_params[..., 3:4][:, None, None, :]
        int_diff = light_params[..., 4:5][:, None, None, :]
        shading = (int_amb + int_diff *
                   torch.clamp(util.dot(light_dir, normal), min=0.0))
        shaded = shading * kd
        return shaded, shading


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


@hydra.main(config_path="../config", config_name="test_fauna")
def main(cfg):
    assert torch.cuda.is_available(), "Cuda is required to run this script."
    resolution = cfg.model.cfg_predictor_base.cfg_shape.grid_res
    resolution = (resolution, resolution)
    render_modes = cfg.render_modes
    if getattr(cfg, "checkpoint_path", None) is not None:
        checkpoint_path = cfg.checkpoint_path
    elif getattr(cfg, "checkpoint_dir", None) is not None and getattr(cfg, "checkpoint_name", None) is not None:
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
    else:
        checkpoints = sorted(
            glob.glob(osp.join(cfg.checkpoint_dir, '*.pth')),
            key=lambda x: int(''.join([c for c in osp.basename(x) if c.isdigit()]))
        )
        checkpoint_path = checkpoints[-1]
    if getattr(cfg, "output_dir") is not None:
        output_dir = cfg.output_dir
    else:
        output_dir = cfg.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)

    batch_size = cfg.dataset.batch_size
    if cfg.finetune_texture:
        batch_size = 1  # finetune texture one by one
        cfg.dataset.batch_size = 1
        cfg.dataset.num_frames = 1
    if "rotation" in render_modes or "canonicalization" in render_modes or "animation" in render_modes:
        batch_size = 1  # render video one by one
        cfg.dataset.batch_size = 1
        cfg.dataset.num_frames = 1
    input_image_dir = cfg.dataset.test_data_dir
    num_frames = 1
    
    # 0. Initialize model.
    device = setup_runtime(cfg)
    model = build_model(cfg.model)
    cp = torch.load(checkpoint_path)
    epoch = cp.get('epoch', 999)
    total_iter = cp.get('total_iter', 999999)
    model.load_model_state(cp)
    model.to(device)

    image_paths = sorted(Path(input_image_dir).rglob("*_rgb.*"))
    save_basenames = [osp.splitext(osp.basename(p))[0].replace('_rgb', "") for p in image_paths]
    total_num = len(image_paths)

    for i in tqdm(range(0, total_num, batch_size)):
        if i > 0: break

        # 1. Load input images.
        images = [pil_loader(p) for p in image_paths[i:min(i + batch_size, total_num + 1)]]
        input_image = torch.stack(
            [torchvision.transforms.ToTensor()(img) for img in images], dim=0).to(device)

        # 1.1. Finetune the model if needed.
        if cfg.finetune_texture:
            if i > 0:
                model.load_model_state(cp)

            num_iters = cfg.finetune_iters
            lr = cfg.finetune_lr
            model.set_train()
            optimizer = torch.optim.Adam(model.netInstance.netTexture.parameters(), lr=lr)

            # Create single-image dataset
            single_data_dir = osp.join(output_dir, "single_image", "0")
            rgb_ext = osp.splitext(osp.basename(image_paths[i]))[1]
            os.makedirs(single_data_dir, exist_ok=True)
            shutil.copyfile(image_paths[i], osp.join(single_data_dir, "0_rgb"+rgb_ext))

            mask_path = str(image_paths[i]).replace("_rgb"+rgb_ext, "_mask.png")
            if osp.isfile(mask_path):
                shutil.copyfile(mask_path, osp.join(single_data_dir, "0_mask.png"))
            else:
                w, h = images[0].size
                mask = Image.fromarray(np.uint8(np.ones((h, w, 3)) * 255))
                mask.save(osp.join(single_data_dir, "0_mask.png"))

            box_path = str(image_paths[i]).replace("_rgb"+rgb_ext, "_box.txt")
            if osp.isfile(box_path):
                shutil.copyfile(box_path, osp.join(single_data_dir, "0_box.txt"))
            else:
                with open(osp.join(single_data_dir, "0_box.txt"), 'w') as f:
                    f.write(f"0 0 0 {w} {h} {w} {h} 0")

            cfg.dataset.load_dino_feature = False
            cfg.dataset.load_dino_cluster = False
            cfg.dataset.data_type = "image"
            cfg.dataset.train_data_dir = osp.dirname(single_data_dir)
            dataloader, _, _ = get_data_loaders(cfg.dataset)

            for iteration in range(num_iters):
                for batch in dataloader:
                    batch = validate_all_to_device(batch, device)
                    
                    prior_shape, dino_pred, bank_embedding = model.netBase(
                        total_iter=total_iter, is_training=False, batch=batch, bank_enc=model.netInstance.netEncoder
                    )
                    shape, pose_raw, pose, mvp, w2c, campos, texture_pred, im_features, deform, all_arti_params, light, forward_aux = \
                        model.netInstance(batch[0], prior_shape, epoch, total_iter, is_training=False)
                    class_vector = bank_embedding[0]
                    shaded, shading, albedo = \
                    model.render(["shaded", "shading", "kd"], shape, texture_pred, mvp, w2c, campos, resolution,
                                 im_features=im_features, light=light, prior_shape=prior_shape,
                                 dino_net=dino_pred, spp=4, num_frames=1, 
                                 class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    
                    image_pred = shaded[:, :3, :, :]
                    image_gt = batch[0].squeeze(1)  # [1, 3, 256, 256]
                    image_loss = 1 * (image_pred - image_gt).abs().mean()
                    optimizer.zero_grad()
                    image_loss.backward()
                    optimizer.step()
                print(f"T{epoch:04}/{iteration:05}")

            shutil.rmtree(single_data_dir, ignore_errors=True)

        with torch.no_grad():
            # 2. Run model and save renderings.
            model.set_eval()

            in_image_size = cfg.dataset.in_image_size
            input_image = torch.nn.functional.interpolate(
                input_image, size=(in_image_size, in_image_size), mode='bilinear', align_corners=False)
            input_image = input_image[:, None, :, :]

            prior_shape, dino_pred, bank_embedding = model.netBase(total_iter=total_iter, is_training=False, 
                                                                   batch=[input_image], bank_enc=model.netInstance.netEncoder)
            shape, pose_raw, pose, mvp, w2c, campos, texture_pred, im_features, deform, all_arti_params, light, forward_aux = \
                model.netInstance(input_image, prior_shape,
                                  epoch, total_iter, is_training=False)
            class_vector = bank_embedding[0]

            if deform is not None:
                deformed_shape = prior_shape.deform(deform)
            else:
                deformed_shape = prior_shape

            if cfg.evaluate_keypoint:
                save_txts(shape.v_pos, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_posed_verts")
                save_txts(pose, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_pose")
                
                v_pos_clip4 = ru.xfm_points(shape.v_pos, mvp)
                v_pos_uv = v_pos_clip4[..., :2] / v_pos_clip4[..., 3:]
                save_txts(v_pos_uv, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_2d_projection_uv")
                
                # Render occlusion
                glctx = dr.RasterizeGLContext()
                v_pos_clip4 = ru.xfm_points(shape.v_pos, mvp)
                v_pos_uv = v_pos_clip4[..., :2] / v_pos_clip4[..., 3:]
                rast, _ = dr.rasterize(
                    glctx, v_pos_clip4, shape.t_pos_idx[0].int(), resolution)
                face_ids = rast[..., -1]
                face_ids = face_ids.view(-1, resolution[0] * resolution[1])
                current_batch, num_verts, _ = shape.v_pos.shape
                res = []
                rendered = []
                dpi = 32
                fx, fy = resolution[1] // dpi, resolution[0] // dpi

                for b in range(current_batch):
                    current_face_ids = face_ids[b]
                    current_face_ids = current_face_ids[current_face_ids > 0]
                    visible_verts = shape.t_pos_idx[0][(
                        current_face_ids - 1).long()].view(-1)
                    visible_verts = torch.unique(visible_verts)
                    visibility = torch.zeros(num_verts, device=device)
                    visibility[visible_verts] = 1
                    res += [visibility]
                    fig = plt.figure(figsize=(fx, fy), dpi=dpi, frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    ax.scatter(v_pos_uv[b, visibility == 1, 0].cpu().numpy(
                    ), v_pos_uv[b, visibility == 1, 1].cpu().numpy(), s=1, c="red")
                    ax.scatter(v_pos_uv[b, visibility == 0, 0].cpu().numpy(
                    ), v_pos_uv[b, visibility == 0, 1].cpu().numpy(), s=1, c="black")
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
                rendered = torch.from_numpy(np.stack(rendered, 0).transpose(0, 3, 1, 2))
                combined = rendered * 0.5 + input_image.cpu().squeeze(1) * 0.5
                res = torch.stack(res, 0)
                save_txts(res, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_binary_occlusion")
                save_images(combined, None, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_2d_projection_image")

                continue

            if "input_view" in render_modes:
                shaded, shading, albedo = \
                    model.render(["shaded", "shading", "kd"], shape, texture_pred, mvp, w2c, campos, resolution,
                                 im_features=im_features, light=light, prior_shape=prior_shape,
                                 dino_net=dino_pred, spp=4, num_frames=1, 
                                 class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                image_pred = shaded[:, :3, :, :]
                mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                shading = shading.expand_as(image_pred)
                save_images(input_image.squeeze(1), None, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_input_image")
                save_images(image_pred, mask_pred, save_basenames[i:min(
                    i+batch_size, total_num + 1)], output_dir, suffix="_input_view_textured")
                if shading is not None:
                    save_images(shading / 2, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_input_view_shading")

                gray_light = FixedDirectionLight(direction=torch.FloatTensor(
                    [0, 0, 1]).to(device), amb=0.2, diff=0.7)
                shaded, shading = \
                    model.render(["shaded", "shading"], shape, texture_pred, mvp, w2c, campos, resolution,
                                 im_features=im_features, light=gray_light, prior_shape=prior_shape,
                                 dino_net=dino_pred, spp=4,
                                 class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                
                mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                shading = shading.expand_as(image_pred)
                if shading is not None:
                    save_images(shading, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_input_view_mesh")

            if "other_views" in render_modes:
                canonical_pose = torch.concat(
                    [torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(device)
                canonical_mvp, canonical_w2c, canonical_campos = model.netInstance.get_camera_extrinsics_from_pose(
                    canonical_pose, offset_extra=5.5)
                
                instance_rotate_angles = [torch.FloatTensor([0, angle, 0]) / 180 * np.pi for angle in range(0, 360, 30)]
                
                gray_light = FixedDirectionLight(direction=torch.FloatTensor(
                    [0, 0, 1]).to(device), amb=0.2, diff=0.7)
                for idx, rot_angle in enumerate(instance_rotate_angles):
                    mtx = torch.eye(4).to(device)
                    mtx[:3, :3] = euler_angles_to_matrix(rot_angle, "XYZ")
                    cur_w2c = torch.matmul(canonical_w2c, mtx[None])
                    cur_mvp = torch.matmul(canonical_mvp, mtx[None])
                    cur_campos = canonical_campos @ torch.linalg.inv(mtx[:3, :3]).T

                    shaded, shading = \
                        model.render(["shaded", "shading"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                    im_features=im_features, light=gray_light, prior_shape=prior_shape,
                                    dino_net=dino_pred, spp=4, num_frames=1,
                                    class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    shading = shading.expand_as(image_pred)
                    save_images(shading, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_other_view_mesh_%d" % idx, mode="transparent")

                    _ = model.render(["shaded"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))

                    ori_light_dir = light.light_params[..., :3]
                    final_dir = torch.matmul(ori_light_dir, w2c[:, :3, :3])
                    final_dir = torch.matmul(
                        final_dir, cur_w2c[:, :3, :3].transpose(2, 1))[:, 0]
                    amb, diff = light.light_params[...,
                                                   3:4], light.light_params[..., 4:5]
                    cur_light = FixedDirectionLight(
                        direction=final_dir, amb=amb, diff=diff)

                    shaded, shading = \
                        model.render(["shaded", "shading"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=cur_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    save_images(image_pred, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_other_view_textured_%d" % idx, mode="transparent")

            if "rotation" in render_modes:
                rotation_angles = np.linspace(0, 360, 75) / 180 * np.pi
                gray_light = FixedDirectionLight(direction=torch.FloatTensor([0, 0, 1]).to(device), amb=0.2, diff=0.7)
                canonical_pose = torch.concat([torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(device)
                canonical_mvp, canonical_w2c, canonical_campos = model.netInstance.get_camera_extrinsics_from_pose(
                    canonical_pose, offset_extra=4)
                
                for idx, angle in enumerate(rotation_angles):
                    rot_angle = torch.FloatTensor([0, angle, 0]).to(device)
                    mtx = torch.eye(4).to(device)
                    mtx[:3, :3] = euler_angles_to_matrix(rot_angle, "XYZ")
                    cur_w2c = torch.matmul(w2c, mtx[None])
                    cur_mvp = torch.matmul(mvp, mtx[None])
                    cur_campos = torch.bmm(torch.linalg.inv(mtx)[None], torch.concat(
                        [campos, torch.ones_like(campos[..., :1])], dim=-1)[..., None]).squeeze(-1)
                    cur_campos = cur_campos[..., :3] / cur_campos[..., 3:]

                    shaded, shading = \
                        model.render(["shaded", "shading"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=gray_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    shading = shading.expand_as(image_pred)
                    save_images(shading, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_{:02d}_rotation_video_mesh".format(idx), mode="white")

                    _ = model.render(["shaded"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))

                    ori_light_dir = light.light_params[..., :3]
                    final_dir = torch.matmul(ori_light_dir, w2c[:, :3, :3])
                    final_dir = torch.matmul(final_dir, cur_w2c[:, :3, :3].transpose(2, 1))[:, 0]
                    amb, diff = light.light_params[..., 3:4], light.light_params[..., 4:5]
                    cur_light = FixedDirectionLight(direction=final_dir, amb=amb, diff=diff)

                    shaded, shading = \
                        model.render(["shaded", "shading"], shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=cur_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    save_images(image_pred, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_{:02d}_rotation_video_textured".format(idx), mode="white")

                rot_imgs = sorted(glob.glob(osp.join(output_dir, "*_rotation_video_mesh.png")))
                rot_imgs_textured = sorted(glob.glob(osp.join(output_dir, "*_rotation_video_textured.png")))
                clip = ImageSequenceClip(rot_imgs, fps=25)
                clip.write_videofile(osp.join(output_dir, save_basenames[i] + "_rotation_mesh.mp4"))
                clip_textured = ImageSequenceClip(rot_imgs_textured, fps=25)
                clip_textured.write_videofile(osp.join(output_dir, save_basenames[i] + "_rotation_textured.mp4"))
                for img in rot_imgs:
                    os.remove(img)
                for img in rot_imgs_textured:
                    os.remove(img)

            if "animation" in render_modes:

                canonical_pose = torch.concat([torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(device)
                canonical_mvp, canonical_w2c, canonical_campos = model.netInstance.get_camera_extrinsics_from_pose(
                    canonical_pose, offset_extra=4)

                viewpoint_arti = torch.FloatTensor([0, -120, 0]) / 180 * np.pi
                mtx = torch.eye(4).to(device)
                mtx[:3, :3] = euler_angles_to_matrix(viewpoint_arti, "XYZ")
                w2c_arti = torch.matmul(canonical_w2c, mtx[None])
                mvp_arti = torch.matmul(canonical_mvp, mtx[None])
                campos_arti = canonical_campos @ torch.linalg.inv(mtx[:3, :3]).T

                deformed_bones, kinematic_tree, _ = estimate_bones(
                    deformed_shape.v_pos[:, None, :, :], model.netInstance.cfg_articulation.num_body_bones,
                    n_legs=model.netInstance.cfg_articulation.num_legs,
                    n_leg_bones=model.netInstance.cfg_articulation.num_leg_bones,
                    body_bones_mode=model.netInstance.cfg_articulation.body_bones_mode, compute_kinematic_chain=True,
                    aux=model.netInstance.bone_aux, attach_legs_to_body=True)
                
                arti_param_files = sorted(glob.glob(osp.join(cfg.arti_param_dir, "arti_params*.txt")))
                arti_params = np.stack([np.loadtxt(f) for f in arti_param_files], axis=0)
                arti_params = arti_params / 180 * np.pi

                interpolate_num = 5
                animate_arti_params = []
                for idx_ in range(0, arti_params.shape[0]):
                    if idx_ == arti_params.shape[0] - 1:
                        break
                    cur_arti_params = arti_params[idx_:idx_+2]
                    animate_arti_params.append(cur_arti_params[0])
                    for j in range(1, interpolate_num):
                        animate_arti_params.append(cur_arti_params[0] * (1 - j / interpolate_num) + cur_arti_params[1] * (j / interpolate_num))
                animate_arti_params.append(arti_params[-1])
                animate_arti_params = np.stack(animate_arti_params, axis=0)
                animate_arti_params = torch.from_numpy(animate_arti_params).to(device)

                gray_light = FixedDirectionLight(direction=torch.FloatTensor([0, 0, 1]).to(device), amb=0.2, diff=0.7)

                num_vis_frames = animate_arti_params.shape[0]

                for arti_id, arti_param in enumerate(animate_arti_params):
                    rot_angle = torch.FloatTensor([0, np.pi * 2 / (num_vis_frames - 1) * arti_id, 0]).to(device)
                    mtx = torch.eye(4).to(device)
                    mtx[:3, :3] = euler_angles_to_matrix(rot_angle, "XYZ")
                    cur_w2c = torch.matmul(w2c_arti, mtx[None])
                    cur_mvp = torch.matmul(mvp_arti, mtx[None])
                    cur_campos = torch.bmm(torch.linalg.inv(mtx)[None], torch.concat(
                        [campos_arti, torch.ones_like(campos_arti[..., :1])], dim=-1)[..., None]).squeeze(-1)
                    cur_campos = cur_campos[..., :3] / cur_campos[..., 3:]

                    arti_param = arti_param[None, None]
                    verts_articulated, aux = skinning(
                        deformed_shape.v_pos, deformed_bones, kinematic_tree, arti_param, output_posed_bones=True,
                        temperature=model.netInstance.cfg_articulation.skinning_temperature)
                    verts_articulated = verts_articulated.squeeze(1)
                    v_tex = prior_shape.v_tex
                    if len(v_tex) != len(verts_articulated):
                        v_tex = v_tex.repeat(len(verts_articulated), 1, 1)
                    posed_shape = make_mesh(
                        verts_articulated,
                        prior_shape.t_pos_idx,
                        v_tex,
                        prior_shape.t_tex_idx,
                        shape.material)

                    _ = model.render(["shaded"], posed_shape, texture_pred, mvp_arti, w2c_arti, campos_arti, resolution,
                                     im_features=im_features, light=light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))

                    ori_light_dir = light.light_params[..., :3]
                    final_dir = torch.matmul(ori_light_dir, w2c[:, :3, :3])
                    final_dir = torch.matmul(final_dir, w2c_arti[:, :3, :3].transpose(2, 1))[:, 0]
                    amb, diff = light.light_params[..., 3:4], light.light_params[..., 4:5]
                    arti_light = FixedDirectionLight(direction=final_dir, amb=amb, diff=diff)

                    shaded, shading = \
                        model.render(["shaded", "shading"], posed_shape, texture_pred, mvp_arti, w2c_arti, campos_arti, resolution,
                                     im_features=im_features, light=arti_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)

                    shaded, shading = \
                        model.render(["shaded", "shading"], posed_shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=arti_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred_rot = shaded[:, :3, :, :]
                    mask_pred_rot = shaded[:, 3:, :, :].expand_as(image_pred_rot)
                    save_images(image_pred, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_{:02d}_animation_video_textured".format(arti_id), mode="white")
                    save_images(image_pred_rot, mask_pred_rot, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_{:02d}_animation_video_textured_rot".format(arti_id), mode="white")
                
                animation_imgs_textured = sorted(
                    glob.glob(osp.join(output_dir, "*_animation_video_textured.png")))
                clip_textured = ImageSequenceClip(
                    animation_imgs_textured, fps=10)
                clip_textured.write_videofile(
                    osp.join(output_dir, save_basenames[i] + "_animation_textured.mp4"), fps=10)
                animation_imgs_textured_rot = sorted(
                    glob.glob(osp.join(output_dir, "*_animation_video_textured_rot.png")))
                clip_textured_rot = ImageSequenceClip(
                    animation_imgs_textured_rot, fps=10)
                clip_textured_rot.write_videofile(osp.join(
                    output_dir, save_basenames[i] + "_animation_textured_rot.mp4"), fps=10)
                for img in animation_imgs_textured:
                    os.remove(img)
                for img in animation_imgs_textured_rot:
                    os.remove(img)

            if "canonicalization" in render_modes:
                if cfg.category != "horse":
                    raise NotImplementedError("Canonicalization mode is only supported for horse.")
                
                num_vis_frames = 25
                
                canon_viewpoint = torch.FloatTensor([0, -120, 0]) / 180 * np.pi
                canon_viewpoint_axis = transforms.matrix_to_axis_angle(
                    transforms.euler_angles_to_matrix(canon_viewpoint, convention="XYZ"))
                
                arti_param_files = sorted(glob.glob(osp.join(cfg.arti_param_dir, "arti_params*.txt")))
                animate_start_arti_param = np.loadtxt(arti_param_files[0]) / 180 * np.pi
                animate_start_arti_param = torch.from_numpy(animate_start_arti_param).to(device).view(1, 20, 3)

                pose_R = pose[:, :9].view(1, 3, 3)
                ori_viewpoint_axis = transforms.matrix_to_axis_angle(pose_R.transpose(-2, -1)).view(3)
                pose_T = pose[:, -3:].view(1, 3, 1)
                starting_arti_param = all_arti_params.clone()

                deformed_bones, kinematic_tree, _ = estimate_bones(
                    deformed_shape.v_pos[:, None], model.netInstance.num_body_bones, n_legs=model.netInstance.num_legs, n_leg_bones=model.netInstance.num_leg_bones,
                    body_bones_mode=model.netInstance.body_bones_mode, compute_kinematic_chain=True, aux=model.netInstance.bone_aux, attach_legs_to_body=True)

                for frame_id in range(num_vis_frames):
                    viewpoint_axis = ori_viewpoint_axis * \
                        (1 - frame_id / (num_vis_frames - 1)) + \
                        canon_viewpoint_axis.to(
                            device) * (frame_id / (num_vis_frames - 1))
                    cur_pose_R = transforms.axis_angle_to_matrix(
                        viewpoint_axis).view(1, 3, 3).to(device).transpose(1, 2)
                    cur_cam_dist = 10 * (1 - frame_id / (num_vis_frames - 1)) + 14 * (frame_id / (num_vis_frames - 1))
                    cur_pose_T = pose_T * (1 - frame_id / (num_vis_frames - 1))
                    cur_pose = torch.cat([cur_pose_R.reshape(1, 9), cur_pose_T.reshape(1, 3)], dim=1)
                    cur_arti_param = starting_arti_param * \
                        (1 - frame_id / (num_vis_frames - 1)) + \
                        animate_start_arti_param * (frame_id / (num_vis_frames - 1))

                    cur_mvp, cur_w2c, cur_campos = model.netInstance.get_camera_extrinsics_from_pose(
                        cur_pose, offset_extra=cur_cam_dist - 10)

                    verts_articulated, aux = skinning(deformed_shape.v_pos, deformed_bones, kinematic_tree, cur_arti_param,
                                                      output_posed_bones=True, temperature=model.netInstance.skinning_temperature)
                    verts_articulated = verts_articulated.squeeze(1)
                    v_tex = prior_shape.v_tex
                    if len(v_tex) != len(verts_articulated):
                        v_tex = v_tex.repeat(len(verts_articulated), 1, 1)
                    posed_shape = make_mesh(
                        verts_articulated,
                        prior_shape.t_pos_idx,
                        v_tex,
                        prior_shape.t_tex_idx,
                        shape.material)
                    current_batch = shape.v_pos.shape[0]

                    _ = model.render(["shaded"], posed_shape, texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))

                    ori_light_dir = light.light_params[..., :3]
                    final_dir = torch.matmul(ori_light_dir, w2c[:, :3, :3])
                    final_dir = torch.matmul(final_dir, cur_w2c[:, :3, :3].transpose(2, 1))[:, 0]
                    amb, diff = light.light_params[..., 3:4], light.light_params[..., 4:5]
                    arti_light = FixedDirectionLight(
                        direction=final_dir, amb=amb, diff=diff)

                    shaded, shading = \
                        model.render(["shaded", "shading"], posed_shape.extend(current_batch), texture_pred, cur_mvp, cur_w2c, cur_campos, resolution,
                                     im_features=im_features, light=arti_light, prior_shape=prior_shape,
                                     dino_net=dino_pred, spp=4,
                                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1))
                    image_pred = shaded[:, :3, :, :]
                    mask_pred = shaded[:, 3:, :, :].expand_as(image_pred)
                    save_images(image_pred, mask_pred, save_basenames[i:min(
                        i+batch_size, total_num + 1)], output_dir, suffix="_{:02d}_canon_video_textured".format(frame_id), mode="white")

                canon_imgs_textured = sorted(glob.glob(osp.join(output_dir, "*_canon_video_textured.png")))
                clip_textured = ImageSequenceClip(canon_imgs_textured, fps=25)
                clip_textured.write_videofile(osp.join(output_dir, save_basenames[i] + "_canon_textured.mp4"), fps=25)
                for img in canon_imgs_textured:
                    os.remove(img)


def validate_tensor_to_device(x, device):
    if torch.any(torch.isnan(x)):
        return None
    else:
        return x.to(device)

def validate_all_to_device(batch, device):
    return tuple(validate_tensor_to_device(x, device) for x in batch)


if __name__ == "__main__":
    main()
