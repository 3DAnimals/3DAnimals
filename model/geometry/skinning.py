import math
import torch
import torch.nn as nn
from . import util
from einops import rearrange


def _joints_to_bones(joints, bones_idxs):
    bones = []
    for a, b in bones_idxs:
        bones += [torch.stack([joints[:, :, a, :], joints[:, :, b, :]], dim=2)]
    bones = torch.stack(bones, dim=2)
    return bones


def _compute_vertices_to_bones_weights(bones_pred, seq_shape_pred, temperature=1):
    vertices_to_bones = []
    for i in range(bones_pred.shape[2]):
        vertices_to_bones += [util.line_segment_distance(bones_pred[:, :, i, 0], bones_pred[:, :, i, 1], seq_shape_pred)]
    # vertices_to_bones = nn.functional.softmax(1 / torch.stack(vertices_to_bones) / temperature, dim=0)
    vertices_to_bones = nn.functional.softmax(-torch.stack(vertices_to_bones) / temperature, dim=0)
    return vertices_to_bones


def build_kinematic_chain(n_bones, start_bone_idx):
    # build bones and kinematic chain starting from leaf bone (body joint)
    bones_to_joints = []
    kinematic_chain = []
    bone_idx = start_bone_idx
    # bones from leaf to root
    dependent_bones = []
    for i in range(n_bones):
        bones_to_joints += [(i + 1, i)]
        kinematic_chain = [(bone_idx, dependent_bones)] + kinematic_chain  # parent is always in the front
        dependent_bones = dependent_bones + [bone_idx]
        bone_idx += 1
    return bones_to_joints, kinematic_chain, dependent_bones


def update_body_kinematic_chain(kinematic_chain, leg_kinematic_chain, body_bone_idx, leg_bone_idxs, attach_legs_to_body=True):
    if attach_legs_to_body:
        for bone_idx, dependent_bones in kinematic_chain:
            if bone_idx == body_bone_idx or body_bone_idx in dependent_bones:
                dependent_bones += leg_bone_idxs
    kinematic_chain =  kinematic_chain + leg_kinematic_chain  # parent is always in the front
    return kinematic_chain


@torch.no_grad()
def estimate_bones(seq_shape, n_body_bones, resample=False, n_legs=4, n_leg_bones=0, body_bones_mode='z_minmax', compute_kinematic_chain=True, aux=None, attach_legs_to_body=True, legs_to_body_joint_indices=None,
                   bone_y_threshold=None):
    """
    Estimate the position and structure of bones given the mesh vertex positions.

    Args:
        seq_shape: a tensor of shape (B, F, V, 3), the batched position of mesh vertices.
        n_body_bones: an integer, the desired number of bones.
    Returns:
        (bones_pred, kinematic_chain) where
        bones_pred: a tensor of shape (B, F, num_bones, 2, 3)
        kinematic_chain: a list of tuples of length n_body_bones; for each tuple, the first element is the bone index while 
                         the second element is a list of bones indices of dependent bones.
    """
    # preprocess shape
    if resample:
        b, _, n, _ = seq_shape.shape
        seq_shape = util.sample_farthest_points(rearrange(seq_shape, 'b f n d -> (b f) d n'), n // 4)
        seq_shape = rearrange(seq_shape, '(b f) d n -> b f n d', b=b)

    if body_bones_mode == 'z_minmax':
        indices_max, indices_min = seq_shape[..., 2].argmax(dim=2), seq_shape[..., 2].argmin(dim=2)
        indices = torch.cat([indices_max[..., None], indices_min[..., None]], dim=2)
        indices_gather = indices[..., None].repeat(1, 1, 1, 3)  # Shape: (B, F, 2, 3)
        points = seq_shape.gather(2, indices_gather)
        point_a = points[:, :, 0, :]
        point_b = points[:, :, 1, :]
    elif body_bones_mode == 'z_minmax_y+':
        mid_point = seq_shape.mean(2)
        seq_shape_pos_y_mask = (seq_shape[:, :, :, 1] > (mid_point[:, :, None, 1] - 0.5)).float()  # y higher than midpoint
        seq_shape_z = seq_shape[:, :, :, 2] * seq_shape_pos_y_mask + (-1e6) * (1 - seq_shape_pos_y_mask)
        indices = seq_shape_z.argmax(2)
        indices_gather = indices[..., None, None].repeat(1, 1, 1, 3)
        point_a = seq_shape.gather(2, indices_gather).squeeze(2)
        seq_shape_z = seq_shape[:, :, :, 2] * seq_shape_pos_y_mask + 1e6 * (1 - seq_shape_pos_y_mask)
        indices = seq_shape_z.argmin(2)
        indices_gather = indices[..., None, None].repeat(1, 1, 1, 3)
        point_b = seq_shape.gather(2, indices_gather).squeeze(2)
    else:
        raise NotImplementedError

    # place points on the symmetry axis
    point_a[..., 0] = 0
    point_b[..., 0] = 0

    mid_point = seq_shape.mean(2)  # Shape: (B, F, 3)
    # place points on the symmetry axis
    mid_point[..., 0] = 0
    if n_leg_bones > 0:
        mid_point[..., 1] += 0.5  # lift mid point a bit higher if there are legs

    assert n_body_bones % 2 == 0
    n_joints = n_body_bones + 1
    blend = torch.linspace(0., 1., math.ceil(n_joints / 2), device=point_a.device)[None, None, :, None]  # Shape: (1, 1, (n_joints + 1) / 2, 1)
    joints_a = point_a[:, :, None, :] * (1 - blend) + mid_point[:, :, None, :] * blend
    # point_a to mid_point
    joints_b = point_b[:, :, None, :] * blend + mid_point[:, :, None, :] * (1 - blend)
    # mid_point to point_b
    joints = torch.cat([joints_a[:, :, :-1], joints_b], 2)  # Shape: (B, F, n_joints, 3)

    # build bones and kinematic chain starting from leaf bones
    if compute_kinematic_chain:
        aux = {}
        half_n_body_bones = n_body_bones // 2
        bones_to_joints = []
        kinematic_chain = []
        bone_idx = 0
        # bones from point_a to mid_point
        dependent_bones = []
        for i in range(half_n_body_bones):
            bones_to_joints += [(i + 1, i)]
            kinematic_chain = [(bone_idx, dependent_bones)] + kinematic_chain  # parent is always in the front
            dependent_bones = dependent_bones + [bone_idx]
            bone_idx += 1
        # bones from point_b to mid_point
        dependent_bones = []
        for i in range(n_body_bones - 1, half_n_body_bones - 1, -1):
            bones_to_joints += [(i, i + 1)]
            kinematic_chain = [(bone_idx, dependent_bones)] + kinematic_chain  # parent is always in the front
            dependent_bones = dependent_bones + [bone_idx]
            bone_idx += 1
        aux['bones_to_joints'] = bones_to_joints
    else:
        bones_to_joints = aux['bones_to_joints']
        kinematic_chain = aux['kinematic_chain']

    bones_pred = _joints_to_bones(joints, bones_to_joints)

    if n_leg_bones > 0:
        assert n_legs == 4
        # attach four legs
        # y, z is symetry plain
        # y axis is up
        #
        # top down view:
        #
        #          |
        #      2   |   1
        #   -------|------ > x
        #      3   |   0
        #          ⌄
        #          z
        #
        # find a point with the lowest y in each quadrant
        # max_dist = (point_a - point_b).norm(p=2, dim=-1)
        xs, ys, zs = seq_shape.unbind(-1)
        if bone_y_threshold is None:
            x_margin = (xs.quantile(0.95) - xs.quantile(0.05)) * 0.2
            quadrant0 = torch.logical_and(xs > x_margin, zs > 0)
            quadrant1 = torch.logical_and(xs > x_margin, zs < 0)
            quadrant2 = torch.logical_and(xs < -x_margin, zs < 0)
            quadrant3 = torch.logical_and(xs < -x_margin, zs > 0)
        
        else:
            y_threshold = ys.quantile(bone_y_threshold)
            flags = (ys < y_threshold)

            x0 = xs[flags].quantile(0.5)
            z0 = zs[flags].quantile(0.5)
            x_margin = (xs[flags].quantile(0.95) - xs[flags].quantile(0.05)) * 0.2
            z_margin = (zs[flags].quantile(0.95) - zs[flags].quantile(0.05)) * 0.2

            quadrant0 = torch.logical_and(xs - x0 > x_margin, zs - z0 > z_margin)
            quadrant1 = torch.logical_and(xs - x0 > x_margin, zs < z0)
            quadrant2 = torch.logical_and(xs - x0 < -x_margin, zs < z0)
            quadrant3 = torch.logical_and(xs - x0 < -x_margin, zs - z0 > z_margin)

        def find_leg_in_quadrant(quadrant, n_bones, body_bone_idx):
            all_joints = torch.zeros([seq_shape.shape[0], seq_shape.shape[1], n_bones + 1, 3], dtype=seq_shape.dtype, device=seq_shape.device)
            for b in range(seq_shape.shape[0]):
                for f in range(seq_shape.shape[1]):
                    # find a point with the lowest y
                    quadrant_points = seq_shape[b, f][quadrant[b, f]]
                    if len(quadrant_points.view(-1)) < 1:
                        import pdb; pdb.set_trace()
                    
                    idx = torch.argmin(quadrant_points[:, 1])  ## lowest y
                    foot = quadrant_points[idx]

                    # find closest point on the body joints (the end joint of the bone)
                    if body_bone_idx is None:
                        # body_bone_idx = int(torch.argmin(torch.norm(bones_pred[b, f, :, 1] - foot[None], dim=1)))
                        body_bone_idx = int(torch.argmin((bones_pred[b, f, :, 1, 2] - foot[None, 2]).abs()))  # closest in z axis
                    body_joint = bones_pred[b, f, body_bone_idx, 1]

                    # create bone structure from the foot to the body joint
                    blend = torch.linspace(0., 1., n_bones + 1, device=seq_shape.device)[:, None]
                    joints = foot[None] * (1 - blend) + body_joint[None] * blend
                    all_joints[b, f] = joints
            return all_joints, body_bone_idx

        quadrants = [quadrant0, quadrant1, quadrant2, quadrant3]
        if legs_to_body_joint_indices is None:
            legs_to_body_joint_indices = [None, None, None, None]
        start_bone_idx = n_body_bones
        all_leg_bones = []
        if compute_kinematic_chain:
            leg_auxs = []
        else:
            leg_auxs = aux['legs']
        for i, quadrant in enumerate(quadrants):
            if compute_kinematic_chain:
                leg_i_aux = {}
                body_bone_idx = legs_to_body_joint_indices[i]
                if i == 2:
                    body_bone_idx = legs_to_body_joint_indices[1]
                elif i == 3:
                    body_bone_idx = legs_to_body_joint_indices[0]

                leg_joints, body_bone_idx = find_leg_in_quadrant(quadrant, n_leg_bones, body_bone_idx=body_bone_idx)
                legs_to_body_joint_indices[i] = body_bone_idx

                leg_bones_to_joints, leg_kinematic_chain, leg_bone_idxs = build_kinematic_chain(n_leg_bones, start_bone_idx=start_bone_idx)
                kinematic_chain = update_body_kinematic_chain(kinematic_chain, leg_kinematic_chain, body_bone_idx, leg_bone_idxs, attach_legs_to_body=attach_legs_to_body)
                leg_i_aux['body_bone_idx'] = body_bone_idx
                leg_i_aux['leg_bones_to_joints'] = leg_bones_to_joints
                start_bone_idx += n_leg_bones
            else:
                leg_i_aux = leg_auxs[i]
                body_bone_idx = leg_i_aux['body_bone_idx']
                leg_joints, _ = find_leg_in_quadrant(quadrant, n_leg_bones, body_bone_idx)
                leg_bones_to_joints = leg_i_aux['leg_bones_to_joints']
            leg_bones = _joints_to_bones(leg_joints, leg_bones_to_joints)
            all_leg_bones += [leg_bones]
            if compute_kinematic_chain:
                leg_auxs += [leg_i_aux]

        all_bones = [bones_pred] + all_leg_bones
        all_bones = torch.cat(all_bones, dim=2)
    else:
        all_bones = bones_pred
    
    if compute_kinematic_chain:
        aux['kinematic_chain'] = kinematic_chain
        if n_leg_bones > 0:
            aux['legs'] = leg_auxs
        return all_bones.detach(), kinematic_chain, aux
    else:
        return all_bones.detach()


def _estimate_bone_rotation(forward):
    """
    (0, 0, 1) = matmul(b, R^(-1))

    assumes y, z is a symmetry plane

    returns R
    """
    forward = nn.functional.normalize(forward, p=2, dim=-1)

    right = torch.FloatTensor([[1, 0, 0]]).to(forward.device)
    right = right.expand_as(forward)
    up = torch.cross(forward, right, dim=-1)
    up = nn.functional.normalize(up, p=2, dim=-1)
    right = torch.cross(up, forward, dim=-1)
    up = nn.functional.normalize(up, p=2, dim=-1)

    R = torch.stack([right, up, forward], dim=-1)

    return R


def children_to_parents(kinematic_tree):
    """
    converts list [(bone1, [children1, ...]), (bone2, [children1, ...]), ...] to [(bone1, [parent1, ...]), ....]
    """
    parents = []
    for bone_id, _ in kinematic_tree:
        # establish a kinematic chain with current bone as the leaf bone
        parents_ids = [parent_id for parent_id, children in kinematic_tree if bone_id in children]
        parents += [(bone_id, parents_ids)]
    return parents


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """ [Borrowed from PyTorch3D]
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """ [Borrowed from PyTorch3D]
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _prepare_transform_mtx(rotation=None, translation=None):
    mtx = torch.eye(4)[None]
    if rotation is not None:
        if len(mtx) != len(rotation):
            assert len(mtx) == 1
            mtx = mtx.repeat(len(rotation), 1, 1)
        mtx = mtx.to(rotation.device)
        mtx[:, :3, :3] = rotation
    if translation is not None:
        if len(mtx) != len(translation):
            assert len(mtx) == 1
            mtx = mtx.repeat(len(translation), 1, 1)
        mtx = mtx.to(translation.device)
        mtx[:, :3, 3] = translation
    return mtx


def _invert_transform_mtx(mtx):
    inv_mtx = torch.eye(4)[None].repeat(len(mtx), 1, 1).to(mtx.device)
    rotation = mtx[:, :3, :3]
    translation = mtx[:, :3, 3]
    inv_mtx[:, :3, :3] = rotation.transpose(1, 2)
    inv_mtx[:, :3, 3] = -torch.bmm(rotation.transpose(1, 2), translation.unsqueeze(-1)).squeeze(-1)
    return inv_mtx


def skinning(v_pos, bones_pred, kinematic_tree, deform_params, output_posed_bones=False, temperature=1):
    """
    """
    device = deform_params.device
    batch_size, num_frames = deform_params.shape[:2]
    shape = v_pos

    # Associate vertices to bones
    vertices_to_bones = _compute_vertices_to_bones_weights(bones_pred, shape.detach(), temperature=temperature)  # Shape: (num_bones, B, F, V)

    rots_pred = deform_params

    # Rotate vertices based on bone assignments
    frame_shape_pred = []
    if output_posed_bones:
        posed_bones = bones_pred.clone()
        if posed_bones.shape[0] != batch_size or posed_bones.shape[1] != num_frames:
            posed_bones = posed_bones.repeat(batch_size, num_frames, 1, 1, 1)  # Shape: (B, F, num_bones, 2, 3)

    # Go through each bone
    for bone_id, _ in kinematic_tree:
        # Establish a kinematic chain with current bone as the leaf bone
        ## TODO: this assumes the parents is always in the front of the list
        parents_ids = [parent_id for parent_id, children in kinematic_tree if bone_id in children]
        chain_ids = parents_ids + [bone_id]
        # Chain from leaf to root
        chain_ids = chain_ids[::-1]

        # Go through the kinematic chain from leaf to root and compose transformation
        transform_mtx = torch.eye(4)[None].to(device)
        for i in chain_ids:
            # Establish transformation
            rest_joint = bones_pred[:, :, i, 0, :].view(-1, 3)
            rest_bone_vector = bones_pred[:, :, i, 1, :] - bones_pred[:, :, i, 0, :]
            rest_bone_rot = _estimate_bone_rotation(rest_bone_vector.view(-1, 3))
            rest_bone_mtx = _prepare_transform_mtx(rotation=rest_bone_rot, translation=rest_joint)
            rest_bone_inv_mtx = _invert_transform_mtx(rest_bone_mtx)

            # Transform to the bone local frame
            transform_mtx = torch.matmul(rest_bone_inv_mtx, transform_mtx)

            # Rotate the mesh in the bone local frame
            rot_pred = rots_pred[:, :, i]
            rot_pred_mat = euler_angles_to_matrix(rot_pred.view(-1, 3), convention='XYZ')
            rot_pred_mtx = _prepare_transform_mtx(rotation=rot_pred_mat, translation=None)
            transform_mtx = torch.matmul(rot_pred_mtx, transform_mtx)

            # Transform to the world frame
            transform_mtx = torch.matmul(rest_bone_mtx, transform_mtx)

        # Transform vertices
        shape4 = rearrange(torch.cat([shape, torch.ones_like(shape[...,:1])], dim=-1), 'b f ... -> (b f) ...')
        seq_shape_bone = torch.matmul(shape4, transform_mtx.transpose(-2, -1))[..., :3]
        seq_shape_bone = rearrange(seq_shape_bone, '(b f) ... -> b f ...', b=batch_size, f=num_frames)

        if output_posed_bones:
            bones4 = torch.cat([rearrange(posed_bones[:, :, bone_id], 'b f ... -> (b f) ...'), torch.ones(batch_size * num_frames, 2, 1).to(device)], dim=-1)
            posed_bones[:, :, bone_id] = rearrange(torch.matmul(bones4, transform_mtx.transpose(-2, -1))[..., :3], '(b f) ... -> b f ...', b=batch_size, f=num_frames)

        # Transform mesh with weights
        frame_shape_pred += [vertices_to_bones[bone_id, ..., None] * seq_shape_bone]

    frame_shape_pred = sum(frame_shape_pred)

    aux = {}
    aux['bones_pred'] = bones_pred
    aux['vertices_to_bones'] = vertices_to_bones
    if output_posed_bones:
        aux['posed_bones'] = posed_bones

    return frame_shape_pred, aux
