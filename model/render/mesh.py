# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch

from . import obj
from . import util

#########################################################################################
# Base mesh class
#
# Minibatch in mesh is supported, as long as each mesh shares the same edge connectivity. 
#########################################################################################
class Mesh:
    def __init__(self, 
                 v_pos=None, 
                 t_pos_idx=None, 
                 v_nrm=None, 
                 t_nrm_idx=None, 
                 v_tex=None, 
                 t_tex_idx=None, 
                 v_tng=None, 
                 t_tng_idx=None, 
                 material=None, 
                 base=None):
        self.v_pos = v_pos
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng
        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx
        self.material = material

        if base is not None:
            self.copy_none(base)

    def __len__(self):
        return len(self.v_pos)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material

    def clone(self):
        out = Mesh(base=self)
        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone().detach()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone().detach()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone().detach()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone().detach()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone().detach()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone().detach()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone().detach()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone().detach()
        return out
    
    def detach(self):
        return self.clone()

    def extend(self, N: int):
        """
        Create new Mesh class which contains each input mesh N times.

        Args:
            N: number of new copies of each mesh.

        Returns:
            new Mesh object.
        """
        verts = self.v_pos.repeat(N, 1, 1)
        faces = self.t_pos_idx
        uvs = self.v_tex.repeat(N, 1, 1)
        uv_idx = self.t_tex_idx
        mat = self.material

        return make_mesh(verts, faces, uvs, uv_idx, self.material)

    def deform(self, deformation):
        """
        Create new Mesh class which is obtained by performing the deformation to the self.

        Args:
            deformation: tensor with shape (B, V, 3)

        Returns:
            new Mesh object after the deformation.
        """
        assert deformation.shape[1] == self.v_pos.shape[1] and deformation.shape[2] == 3
        verts = self.v_pos + deformation
        return make_mesh(verts, self.t_pos_idx, self.v_tex.repeat(len(verts), 1, 1), self.t_tex_idx, self.material)

    def get_m_to_n(self, m: int, n: int):
        """
        Create new Mesh class with the n-th (included) mesh to the m-th (not included) mesh in the batch.

        Args:
            m: the index of the starting mesh to be contained.
            n: the index of the first mesh not to be contained.
        """
        verts = self.v_pos[m:n, ...]
        faces = self.t_pos_idx
        uvs = self.v_tex[m:n, ...]
        uv_idx = self.t_tex_idx
        mat = self.material

        return make_mesh(verts, faces, uvs, uv_idx, mat)

    def first_n(self, n: int):
        """
        Create new Mesh class with only the first n meshes in the batch.

        Args:
            n: number of meshes to be contained.

        Returns:
            new Mesh object with the first n meshes.
        """
        return self.get_m_to_n(0, n)
        verts = self.v_pos[:n, ...]
        faces = self.t_pos_idx
        uvs = self.v_tex[:n, ...]
        uv_idx = self.t_tex_idx
        mat = self.material

        return make_mesh(verts, faces, uvs, uv_idx, mat)

    def get_n(self, n: int):
        """
        Create new Mesh class with only the n-th meshes in the batch.

        Args:
            n: the index of the mesh to be contained.

        Returns:
            new Mesh object with the n-th mesh.
        """
        verts = self.v_pos[n:n+1, ...]
        faces = self.t_pos_idx
        uvs = self.v_tex[n:n+1, ...]
        uv_idx = self.t_tex_idx
        mat = self.material

        return make_mesh(verts, faces, uvs, uv_idx, mat)


######################################################################################
# Mesh loading helper
######################################################################################
def load_mesh(filename, mtl_override=None):
    name, ext = os.path.splitext(filename)
    if ext == ".obj":
        return obj.load_obj(filename, clear_ks=True, mtl_override=mtl_override)
    assert False, "Invalid mesh file extension"

######################################################################################
# Compute AABB
######################################################################################
def aabb(mesh):
    return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values

######################################################################################
# Compute unique edge list from attribute/vertex index list
######################################################################################
def compute_edges(attr_idx, return_inverse=False):
    with torch.no_grad():
        # Create all edges, packed by triangle
        idx = attr_idx[0]
        all_edges = torch.cat((
            torch.stack((idx[:, 0], idx[:, 1]), dim=-1),
            torch.stack((idx[:, 1], idx[:, 2]), dim=-1),
            torch.stack((idx[:, 2], idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Eliminate duplicates and return inverse mapping
        return torch.unique(sorted_edges, dim=0, return_inverse=return_inverse)

######################################################################################
# Compute unique edge to face mapping from attribute/vertex index list
######################################################################################
def compute_edge_to_face_mapping(attr_idx, return_inverse=False):
    with torch.no_grad():
        # Get unique edges
        # Create all edges, packed by triangle
        idx = attr_idx[0]
        all_edges = torch.cat((
            torch.stack((idx[:, 0], idx[:, 1]), dim=-1),
            torch.stack((idx[:, 1], idx[:, 2]), dim=-1),
            torch.stack((idx[:, 2], idx[:, 0]), dim=-1),
        ), dim=-1).view(-1, 2)

        # Swap edge order so min index is always first
        order = (all_edges[:, 0] > all_edges[:, 1]).long().unsqueeze(dim=1)
        sorted_edges = torch.cat((
            torch.gather(all_edges, 1, order),
            torch.gather(all_edges, 1, 1 - order)
        ), dim=-1)

        # Elliminate duplicates and return inverse mapping
        unique_edges, idx_map = torch.unique(sorted_edges, dim=0, return_inverse=True)

        tris = torch.arange(idx.shape[0]).repeat_interleave(3).cuda()

        tris_per_edge = torch.zeros((unique_edges.shape[0], 2), dtype=torch.int64).cuda()

        # Compute edge to face table
        mask0 = order[:,0] == 0
        mask1 = order[:,0] == 1
        tris_per_edge[idx_map[mask0], 0] = tris[mask0]
        tris_per_edge[idx_map[mask1], 1] = tris[mask1]

        return tris_per_edge

######################################################################################
# Align base mesh to reference mesh:move & rescale to match bounding boxes.
######################################################################################
def unit_size(mesh):
    with torch.no_grad():
        vmin, vmax = aabb(mesh)
        scale = 2 / torch.max(vmax - vmin).item()
        v_pos = mesh.v_pos - (vmax + vmin) / 2 # Center mesh on origin
        v_pos = v_pos * scale                  # Rescale to unit size

        return Mesh(v_pos, base=mesh)

######################################################################################
# Center & scale mesh for rendering
######################################################################################
def center_by_reference(base_mesh, ref_aabb, scale):
    center = (ref_aabb[0] + ref_aabb[1]) * 0.5
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item()
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale
    return Mesh(v_pos, base=base_mesh)

######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(imesh):
    batch_size = imesh.v_pos.shape[0]

    i0 = imesh.t_pos_idx[0, :, 0]  # Shape: (F)
    i1 = imesh.t_pos_idx[0, :, 1]  # Shape: (F)
    i2 = imesh.t_pos_idx[0, :, 2]  # Shape: (F)

    v0 = imesh.v_pos[:, i0, :]  # Shape: (B, F, 3)
    v1 = imesh.v_pos[:, i1, :]  # Shape: (B, F, 3)
    v2 = imesh.v_pos[:, i2, :]  # Shape: (B, F, 3)

    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # Shape: (B, F, 3)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(imesh.v_pos)  # Shape: (B, V, 3)
    v_nrm.scatter_add_(1, i0[None, :, None].repeat(batch_size, 1, 3), face_normals)
    v_nrm.scatter_add_(1, i1[None, :, None].repeat(batch_size, 1, 3), face_normals)
    v_nrm.scatter_add_(1, i2[None, :, None].repeat(batch_size, 1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(util.dot(v_nrm, v_nrm) > 1e-20, 
                        v_nrm, torch.tensor([0.0, 0.0, 1.0], 
                        dtype=torch.float32, device='cuda'))
    v_nrm = util.safe_normalize(v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(v_nrm))

    return Mesh(v_nrm=v_nrm, t_nrm_idx=imesh.t_pos_idx, base=imesh)

######################################################################################
# Compute tangent space from texture map coordinates
# Follows http://www.mikktspace.com/ conventions
######################################################################################
def compute_tangents(imesh):
    batch_size = imesh.v_pos.shape[0]

    vn_idx = [None] * 3
    pos = [None] * 3
    tex = [None] * 3
    for i in range(0,3):
        pos[i] = imesh.v_pos[:, imesh.t_pos_idx[0, :, i]]
        tex[i] = imesh.v_tex[:, imesh.t_tex_idx[0, :, i]]
        vn_idx[i] = imesh.t_nrm_idx[..., i:i+1]

    tangents = torch.zeros_like(imesh.v_nrm)
    tansum   = torch.zeros_like(imesh.v_nrm)

    # Compute tangent space for each triangle
    uve1 = tex[1] - tex[0]  # Shape: (B, F, 2)
    uve2 = tex[2] - tex[0]  # Shape: (B, F, 2)
    pe1  = pos[1] - pos[0]  # Shape: (B, F, 3)
    pe2  = pos[2] - pos[0]  # Shape: (B, F, 3)
    
    nom   = pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2]  # Shape: (B, F, 3)
    denom = uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1]  # Shape: (B, F, 1)
    
    # Avoid division by zero for degenerated texture coordinates
    tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))  # Shape: (B, F, 3)

    # Update all 3 vertices
    for i in range(0,3):
        idx = vn_idx[i].repeat(batch_size, 1, 3)  # Shape: (B, F, 3)
        tangents.scatter_add_(1, idx, tang)       # tangents[n_i] = tangents[n_i] + tang
        tansum.scatter_add_(1, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
    tangents = tangents / tansum

    # Normalize and make sure tangent is perpendicular to normal
    tangents = util.safe_normalize(tangents)
    tangents = util.safe_normalize(tangents - util.dot(tangents, imesh.v_nrm) * imesh.v_nrm)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(tangents))

    return Mesh(v_tng=tangents, t_tng_idx=imesh.t_nrm_idx, base=imesh)

######################################################################################
# Create new Mesh from verts, faces, uvs, and uv_idx. The rest is auto computed.
######################################################################################
def make_mesh(verts, faces, uvs, uv_idx, material):
    """
    Create new Mesh class with given verts, faces, uvs, and uv_idx.

    Args:
        verts: tensor of shape (B, V, 3)
        faces: tensor of shape (1, F, 3)
        uvs: tensor of shape (B, V, 2)
        uv_idx: tensor of shape (1, F, 3)
        material: an Material instance, specifying the material of the mesh.

    Returns:
        new Mesh object.
    """
    assert len(verts.shape) == 3 and len(faces.shape) == 3 and len(uvs.shape) == 3 and len(uv_idx.shape) == 3, "All components must be batched."
    assert faces.shape[0] == 1 and uv_idx.shape[0] == 1, "Every mesh must share the same edge connectivity."
    assert verts.shape[0] == uvs.shape[0], "Batch size must be consistent."
    ret = Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)
    ret = auto_normals(ret)
    ret = compute_tangents(ret)
    return ret
