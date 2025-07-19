#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, density_to_alpha, alpha_to_density
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud, BasicTetrahedra
from utils.general_utils import strip_symmetric, build_scaling_rotation, inverse_strip
from scene.inn_model import DeformNetwork
from scene.nvp_model import NVPSimplified
# from tetranerf.utils.extension import TetrahedraTracer
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
     

    """
    Ablation study of TetrhedraModel

    1. density的激活函数 # softplus
    2. density转alpha时的尺度计算  
    3. weights的计算方式
    4. weights,vertices转四面体的方式
    5. 分裂的方式
    6. clone的方式

    Returns:
        _type_: _description_
    """
    
def save_points_as_ply(point: torch.Tensor,path):
    import numpy as np
    import open3d as o3d
    points = point.cpu().numpy()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, point_cloud)
    
def soft_weight(weight):
    
    return torch.exp(weight) / torch.sum(torch.exp(weight), dim=-2, keepdim=True)
      
def check_tensor(tensor, name="Tensor"):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values!")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values!")
    
class TetrahedraModel:

    def setup_functions(self):
        
        def build_mean_covariance_from_weighted_vertices(vertices, weight, scaling_modifier = 1.0, cov_modifier = 1e-9):
            """

            Args:
                vertices (_torch.tensor_): [T,4,3]
                weight (_torch.tensor_): [T,4,1]
                scaling_modifier (float, optional): _description_. Defaults to 1.0.

            Returns:
                _type_: _description_
            """
            assert vertices.shape[1] == 4, "Vertices must be of shape (N, 4, 3)"
            assert weight.shape[1] == 4, "Weights must be of shape (N, 4, 1)"
            assert weight.shape[2] == 1, "Weights must be of shape (N, 4, 1)"
            
            mean = torch.sum(vertices * weight, dim=1) / torch.sum(weight, dim=1)
            
            diff = (vertices - mean[:,None,:]).unsqueeze(-1)
            
            # cov = torch.sum(weight[...,None] * diff @ diff.transpose(-1, -2), dim=1) / torch.sum(weight, dim=1)[...,None] * scaling_modifier + cov_modifier * torch.eye(3, device=vertices.device)
            
            cov = torch.sum(weight[...,None] * diff @ diff.transpose(-1, -2), dim=1)  * scaling_modifier + cov_modifier * torch.eye(3, device=vertices.device)
            
                        
            # cov = torch.sum(diff @ diff.transpose(-1, -2), dim=1)  * scaling_modifier + cov_modifier * torch.eye(3, device=vertices.device)
            
            return mean, strip_symmetric(cov)

        
        # def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        #     L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        #     actual_covariance = L @ L.transpose(1, 2)
        #     symm = strip_symmetric(actual_covariance)
        #     return symm
        

        self.density_activation = torch.relu
        self.inverse_density_activation = torch.relu
        self.opacity_activation = torch.relu
        self.inverse_opacity_activation = torch.relu
        self.mean_covariance_activation = build_mean_covariance_from_weighted_vertices
        self.scale_activation = torch.exp
        self.inverse_scale = torch.log
        self.cell_opacity_activation = torch.sigmoid
        self.inverse_cell_opacity = inverse_sigmoid
        self.freeze = False
  
        
        # self.opacity_activation = torch.sigmoid
        # self.inverse_opacity_activation = inverse_sigmoid

        # self.rotation_activation = torch.nn.functional.normalize
    
    def setup_model(self, opt):
        if opt.model_type == "NDR":
            self.inn_network = DeformNetwork(d_feature=opt.embedding_dim,
                                        scale_included=True,
                                        n_blocks=opt.n_blocks,
                                        d_hidden=opt.hidden_dim,
                                        n_layers=opt.n_layers,
                                        skip_in=opt.skip_in,
                                        multires=opt.multires,
                                        weight_norm=opt.weight_norm,).to(self._xyz.device)
        elif opt.model_type == "NVP":
            self.inn_network = NVPSimplified(n_layers=opt.n_blocks,
                                            feature_dims=opt.embedding_dim,
                                            hidden_size=opt.hidden_size,
                                            proj_dims=opt.proj_dims,
                                            code_proj_hidden_size=[],
                                            proj_type=opt.proj_type,
                                            pe_freq=opt.pe_freq,
                                            normalization=opt.normalization,
                                            activation=nn.LeakyReLU,
                                            device=self._xyz.device).to(self._xyz.device)
            
        self.deformation_code = nn.Parameter(torch.zeros(opt.embedding_dim, device=self._xyz.device))


    def __init__(self, sh_degree, scaling_modifier=2.0,use_opacity=True,use_cell_opacity=True,optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._density = torch.empty(0)
        self._cells = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scale = torch.empty(0)
        self._cell_o = torch.empty(0)
        # self._scaling = torch.empty(0)
        # self._rotation = torch.empty(0)
        # self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.cell_gradient_accum = torch.empty(0)
        self.cell_gradient_accum_abs = torch.empty(0)
        self._init_volume = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.scaling_modifier = scaling_modifier
        self.alpha_ratio = 1.0
        self.use_opacity = use_opacity
        self.use_cell_opacity = use_cell_opacity
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self.get_xyz(),
            self._features_dc,
            self._features_rest,
            self._opacity,
            # self.inn_network,
            # self.deformation_code,
            self._cell_o,
            self._scale,
            self._cells,
            self.max_radii2D,
            self.freeze,
            self.cell_gradient_accum,
            self.cell_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.use_opacity,
            self.use_cell_opacity
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._opacity,
        self._cell_o,
        self._scale,
        self._cells,
        self.max_radii2D, 
        self.freeze,
        cell_gradient_accum, 
        cell_gradient_accum_abs, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.use_opacity,
        self.use_cell_opacity) = model_args
        # self.training_setup(training_args)
        self.cell_gradient_accum = cell_gradient_accum
        self.cell_gradient_accum_abs = cell_gradient_accum_abs
        self.denom = denom
        # self.optimizer.load_state_dict(opt_dict)
        self.freeze = True

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    
    
    def get_xyz(self):
        if not self.freeze:
            deformed_xyz = self.inn_network(self.deformation_code, self._xyz)
        else:
            deformed_xyz = self._xyz
        # deformed_xyz = self._xyz
        return deformed_xyz
    
    @property
    def get_cell_scale(self):
        return self.scale_activation(self._scale)
    
    @property
    def get_cell_opacity(self):
        return self.cell_opacity_activation(self._cell_o)
    
    @property
    def get_cells(self):
        return self._cells
    
    @property
    def num_vertices(self):
        return self._xyz.shape[0]
    
    @property
    def num_cells(self):
        return self._cells.shape[0]
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_tetra_v(self):
        return self.get_xyz()[self._cells]
    
    @property
    def get_density(self):
        return self.density_activation(self._density)
    

    
    def get_max_edge(self):
        return torch.stack([self.get_tetra_v[:,0] - self.get_tetra_v[:,1],
                          self.get_tetra_v[:,0] - self.get_tetra_v[:,2],
                          self.get_tetra_v[:,0] - self.get_tetra_v[:,3],
                          self.get_tetra_v[:,1] - self.get_tetra_v[:,2],
                          self.get_tetra_v[:,1] - self.get_tetra_v[:,3],
                          self.get_tetra_v[:,2] - self.get_tetra_v[:,3]], dim=1).norm(dim=-1).max(dim=-1)[0]
    
    def get_cell_aabb(self):
        # [T, 2, 3] min:max
        return torch.stack([self.get_tetra_v.min(dim=-2)[0], self.get_tetra_v.max(dim=-2)[0]], dim=1)
    
    
    def points_in_tetrahedra(self,points):
        # points: (N,3)
        cell_aabb = self.get_cell_aabb() # [T, 2, 3]
        
        # points in cell aabb mask
        points_mask_coarse = ((points[None,:] > cell_aabb[:,0].unsqueeze(1)) & (points[None,:] < cell_aabb[:,1].unsqueeze(1))).all(dim=-1) # [T, N, 3]
        selected_tetra_idx,selected_points_idx = points_mask_coarse.nonzero(as_tuple=True)
        selected_points = points[selected_points_idx]  # [M, 3]
        selected_tetrahedra = self.get_tetra_v[selected_tetra_idx]  # [M, 4, 3]
        

        V0 = selected_tetrahedra[:, 0, :]  # [M, 3]
        V1 = selected_tetrahedra[:, 1, :]  # [M, 3]
        V2 = selected_tetrahedra[:, 2, :]  # [M, 3]
        V3 = selected_tetrahedra[:, 3, :]  # [M, 3]


        M = torch.stack([V1 - V0, V2 - V0, V3 - V0], dim=1)  # [M, 3, 3]
        M_inv = torch.inverse(M)  # [M, 3, 3]


        P_minus_V0 = selected_points - V0  # [M, 3]

   
        bary_coords = torch.einsum('mc,mcd->md', P_minus_V0, M_inv)  # [M, 3]
        alpha = bary_coords[:, 0]  # [M]
        beta = bary_coords[:, 1]   # [M]
        gamma = bary_coords[:, 2]  # [M]
        delta = 1 - alpha - beta - gamma  # [M]


        mask_inside = (alpha >= 0) & (beta >= 0) & (gamma >= 0) & (delta >= 0)  # [M]
        
        result = torch.zeros(points.shape[0], dtype=torch.bool)  # [N]
        
        result[selected_points_idx[mask_inside]] = True
        
        return result

    @staticmethod
    def get_scale(tetra):
        
        """
        Args: 
            tetra: torch.tensor of shape (T, 4, 3)
        Returns:
            scale: torch.tensor of shape (T, 1)
        """
        return TetrahedraModel.volume_of_tetrahedra(tetra) ** (1/3)
    # TODO: detach() is used to avoid gradients, but it may be necessary to remove it

    def get_weights(self):
        
        if self.use_opacity or self.use_cell_opacity:
            return self.opacity_activation(self._opacity)[self._cells]
        else:
            return density_to_alpha(self.get_density[self._cells],self.get_scale(self.get_xyz()[self._cells])[:,None,:].repeat(1,4,1)).clamp(min=1e-5,max=1-1e-5)
        
    
    # def get_covariance(self):
    #     if self.use_cell_opacity:
    #         return self.covariance_activation(self.get_xyz()[self._cells], self.get_weights(), scaling_modifier=self.get_cell_scale) ### TODO
    #     else:
    #         return self.covariance_activation(self.get_xyz()[self._cells], self.get_weights(), scaling_modifier=self.scaling_modifier) ### TODO
    
    def convert_gaussian(self):
        """
           convert tetrahedra to gaussians
        """
        # density = self.get_density[self._cells] # (T,4,1)
        # # print(density.max())
        # scales = self.get_scale(self.get_xyz()[self._cells])[:,None].repe
        # at(1,4,1) # (T,4,1)
        # weights = density_to_alpha(density, scales).clamp(min=0.00001,max=0.99999) # (T,4,1)
        weights = self.get_weights().clamp(min=0.00001,max=30.0)
        vertices = self.get_xyz()[self._cells] # (T,4,3)
        
        mean,cov3d = self.mean_covariance_activation(vertices, weights, scaling_modifier=self.get_cell_scale[...,None])
        # print(self.use_cell_opacity)
        # print(self.use_opacity)
        if self.use_cell_opacity:
            opacity = self.get_cell_opacity # (T,1)   # TODO
        else:
            # opacity = 1 - torch.prod(1-weights,dim=-2)**(1/4) # (T,1)   # TODO
            opacity = (weights * weights / weights.sum(dim=-2)[...,None]).sum(dim=-2)
        # feature_dc = self._features_dc[self._cells] # (T,4,3)
        # feature_rest = self._features_rest[self._cells] # (T,4,3)
        
            
        # 检查 density
        # if torch.isnan(density).any():
        #     raise ValueError("density contains NaN values!")
        # if torch.isinf(density).any():
        #     raise ValueError("density contains Inf values!")

        # # 检查 scales
        # if torch.isnan(scales).any():           
        #     raise ValueError("scales contains NaN values!")
        # if torch.isinf(scales).any():
        #     raise ValueError("scales contains Inf values!")

        # # 检查 weights
        # if torch.isnan(weights).any():
        #     raise ValueError("weights contains NaN values!")
        # if torch.isinf(weights).any():
        #     raise ValueError("weights contains Inf values!")

        # # 检查 vertices
        # if torch.isnan(vertices).any():
        #     raise ValueError("vertices contains NaN values!")
        # if torch.isinf(vertices).any():
        #     raise ValueError("vertices contains Inf values!")

        # # 检查 mean
        # if torch.isnan(mean).any():
        #     raise ValueError("mean contains NaN values!")
        # if torch.isinf(mean).any():
        #     raise ValueError("mean contains Inf values!")

        # # 检查 cov3d
        # if torch.isnan(cov3d).any():
        #     raise ValueError("cov3d contains NaN values!")
        # if torch.isinf(cov3d).any():
        #     raise ValueError("cov3d contains Inf values!")

        # # 检查 opacity
        # if torch.isnan(opacity).any():
        #     raise ValueError("opacity contains NaN values!")
        # if torch.isinf(opacity).any():
        #     raise ValueError("opacity contains Inf values!")
        return mean, cov3d, opacity, weights
        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def unfreeze_inn(self):
        for param in self.inn_network.parameters():
            param.requires_grad = True
        self.freeze = False
            
    def freeze_inn(self):
        for param in self.inn_network.parameters():
            param.requires_grad = False
        self.freeze = True

    
    def reset_inn(self,training_args):
        # create a new inn_network
        self.setup_model()
        self.training_setup(training_args)
            
    def update_cells_and_faces(self):
        self.boundary = self.get_boundary_faces_mask() # (T,1)
        
    def get_boundary_faces_mask(self) -> torch.Tensor:
        """
        判断每个四面体的每个面是否为边界面。
        边界面定义为只被一个四面体包含的面。

        Returns:
            torch.Tensor: 形状为(T,4)的布尔张量,表示每个四面体的4个面是否为边界面
        """
        T = self._cells.shape[0]
        device = self._cells.device

        # 获取每个四面体的4个面
        faces0 = self._cells[:, [0, 1, 2]]  # 面0: [v0,v1,v2] 
        faces1 = self._cells[:, [0, 2, 3]]  # 面1: [v0,v2,v3]
        faces2 = self._cells[:, [0, 3, 1]]  # 面2: [v0,v3,v1]
        faces3 = self._cells[:, [1, 3, 2]]  # 面3: [v1,v3,v2]

        # 将所有面堆叠为一个大张量 (T*4, 3)
        all_faces = torch.cat([faces0, faces1, faces2, faces3], dim=0)

        # 对每个面内的顶点排序以保证一致性
        sorted_faces, _ = torch.sort(all_faces, dim=1)

        # 找出所有唯一面及其出现次数
        _, inverse_indices, counts = torch.unique(sorted_faces, return_inverse=True, return_counts=True, dim=0)

        # 边界面是只出现一次的面
        boundary_mask = counts[inverse_indices] == 1  # (T*4,)

        # 重塑为(T,4)形状,每行对应一个四面体的4个面
        boundary_mask = boundary_mask.view(T, 4)

        return boundary_mask
    

             
    def create_from_tetra(self, tetra : BasicTetrahedra, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_xyz = torch.tensor(np.asarray(tetra.vertices)).float().cuda()
        if tetra.colors is not None:
            fused_color = RGB2SH(torch.tensor(np.asarray(tetra.colors)).float().cuda())
        else:
            fused_color = torch.rand((fused_xyz.shape[0], 3), device="cuda")
        features = torch.zeros((fused_xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0
        
        self._xyz = nn.Parameter(fused_xyz.contiguous().requires_grad_(False))
        print("Number of points at initialisation : ", fused_xyz.shape[0])
        self._cells = torch.tensor(np.asarray(tetra.cells)).long().cuda()
        
        # self.update_signed_volume()
        self.posotive_signed_volume(self._xyz)
        # set opacities around 0.1
        tet_scale = self.get_scale(self._xyz[self._cells])

        # set scale around 1
        init_cell_scale = self.inverse_scale(torch.ones_like(tet_scale))
        self._scale = nn.Parameter(init_cell_scale.contiguous().requires_grad_(True))

        # computer average scale of each vertices
        vertex_indices = self._cells.view(-1).long()  # [T*4]
        scales_repeated = tet_scale.view(-1).repeat_interleave(4)  # [T*4]
        vertex_scale_sum = torch.zeros(self.num_vertices, device=self._xyz.device).scatter_add_(0, vertex_indices, scales_repeated) # [N]
        vertex_scale_count = torch.zeros(self.num_vertices, device=self._xyz.device).scatter_add_(0, vertex_indices, torch.ones_like(scales_repeated)) # [N]
        average_scale = vertex_scale_sum / vertex_scale_count.clamp(min=1) # [N]
        if self.use_opacity or self.use_cell_opacity:
            init_opacity = self.inverse_opacity_activation(torch.ones_like(average_scale)[...,None] * 0.1)
            self._opacity = nn.Parameter(init_opacity.contiguous().requires_grad_(True))
            init_cell_opacity = self.inverse_opacity_activation(torch.ones_like(tet_scale) * 0.1)
            self._cell_o = nn.Parameter(init_cell_opacity.contiguous().requires_grad_(True))
        else:
            init_densities = self.inverse_density_activation(alpha_to_density(torch.ones_like(average_scale)[...,None] * 0.1,average_scale[...,None]))
            self._density = nn.Parameter(init_densities.contiguous().requires_grad_(True))

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1,2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1,2).contiguous().requires_grad_(True))
        
        self.update_cells_and_faces()
        self.max_radii2D = torch.zeros((self.get_cells.shape[0]), device="cuda")
    
    @torch.no_grad()
    def update_signed_volume(self):
        self._init_volume = self.signed_volume_of_tetra()
        swap_mask = self._init_volume < 0 
        self._cells[swap_mask] = self._cells[swap_mask][:,[0,1,3,2]]
         
    def posotive_signed_volume(self, xyz):

        # Calculate the signed volume of tetrahedra
        # v01, v02, v03 are three edges from vertex v0 to v1, v2, v3 respectively
        # The signed volume is 1/6 of the determinant of the matrix formed by these three edges
        # If volume is negative, swap vertices to make it positive
        
        v0 = xyz[self._cells[:, 0]]
        v1 = xyz[self._cells[:, 1]]
        v2 = xyz[self._cells[:, 2]]
        v3 = xyz[self._cells[:, 3]]
        

        v01 = v1 - v0
        v02 = v2 - v0
        v03 = v3 - v0

        volume = torch.det(torch.stack([v01, v02, v03], dim=-1))


        negative_volume_mask = volume < 0
        if negative_volume_mask.any():
            self._cells[negative_volume_mask] = self._cells[negative_volume_mask][:, [1, 0, 2, 3]]
        


    # def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    #     scales = torch.sqrt(dist2*3)[...,None]
    #     tetrahedra_vertices = self.generate_init_tetrahedra(fused_point_cloud, scales).reshape(-1,3)
        
    #     if self.use_opacity:
    #         tetrahedra_opacity = self.inverse_opacity_activation(0.1*torch.ones((tetrahedra_vertices.shape[0],1),device=tetrahedra_vertices.device,dtype=tetrahedra_vertices.dtype))
    #     else:
    #         tetrahedra_density = self.inverse_density_activation(alpha_to_density(torch.ones_like(scales).repeat_interleave(4,dim=0)*0.1,scales.repeat_interleave(4, dim=0))[...,None]).squeeze(-1)
    #     self._xyz = nn.Parameter(tetrahedra_vertices.contiguous().requires_grad_(False))
    #     self._cells =  torch.arange(0, tetrahedra_vertices.shape[0], 1, 
    #                                              device="cuda",
    #                                              dtype=torch.int32).long().reshape(-1, 4).contiguous()
    #     if self.use_opacity:
    #         self._opacity = nn.Parameter(tetrahedra_opacity.contiguous().requires_grad_(True))
    #     else:
    #         self._density = nn.Parameter(tetrahedra_density.contiguous().requires_grad_(True))
    #     # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     # rots[:, 0] = 1

    #     # opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     # self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:,:,0:1].repeat_interleave(4,dim=0).transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:,:,1:].repeat_interleave(4,dim=0).transpose(1, 2).contiguous().requires_grad_(True))
    #     # self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     # self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     # self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros((self.get_cells.shape[0]), device="cuda")
           
    def signed_volume_of_tetra(self):
        vertices = self.get_tetra_v
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]
        d = vertices[:, 3]
        return torch.det(torch.stack([b-a, c-a, d-a], dim=1))



    def regloss_of_signed_volume(self,volume=True,condition_number=True,barrier=False,epsilon=1e-4,s_j=0.5):
        volume = self.signed_volume_of_tetra() /(2 * self._init_volume)
        full_volume = volume + torch.sqrt(volume ** 2 )
        barrier_func = lambda x : 1 /((x**3/s_j**3 - 3*x**2/s_j**2 + 3*x/s_j) +  epsilon)
        return torch.mean(barrier_func(full_volume))
    
    def regloss_of_deltav(self):
        return ((self._xyz - self.get_xyz()).norm(dim=-1,p=2)).norm()
    
    # def regloss_of_edge(self):
    #     vertices = self._xyz
    #     a = vertices[:, 0]
    #     b = vertices[:, 1]
    #     c = vertices[:, 2]
    #     d = vertices[:, 3]
    #     original_edge = torch.stack([b-a,c-a,d-a,c-b,d-b,d-c],dim=-2)

        
        
    def regloss_of_volume(self):
        vertices = self._xyz[self._cells]
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]
        d = vertices[:, 3]
        original_volume = torch.det(torch.stack([b-a, c-a, d-a], dim=1))
        now_volume = self.signed_volume_of_tetra()
        return torch.abs((now_volume - original_volume)).mean()
    
    @staticmethod
    def volume_of_tetrahedra(vertices):
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]
        d = vertices[:, 3]
        return torch.abs(torch.det(torch.stack([a-b, a-c, a-d], dim=1))).unsqueeze(-1)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.cell_gradient_accum = torch.zeros((self.get_cells.shape[0], 1), device="cuda")
        self.cell_gradient_accum_abs = torch.zeros((self.get_cells.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_cells.shape[0], 1), device="cuda")
        self.use_opacity = training_args.use_opacity

        
        l = [ 
           #  {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # 添加 deformation_code 和 inn_network 的参数
            {'params': [self.deformation_code], 'lr': training_args.deformation_code_lr_init * self.spatial_lr_scale, "name": "deformation_code"},
            {'params': self.inn_network.parameters(), 'lr': training_args.inn_lr_init * self.spatial_lr_scale, "name": "inn_network"},
        ]
        
        l.append({'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "density"})
        if self.use_cell_opacity :
            l.append({'params': [self._cell_o], 'lr': training_args.cell_opacity_lr, "name": "cell_opacity"})
            l.append({'params': [self._scale], 'lr': training_args.scale_lr, "name": "scale"})
            
        if not self.use_cell_opacity and not self.use_opacity:
            l.append({'params': [self._density], 'lr': training_args.opacity_lr, "name": "density"})
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
                # 为 deformation_code 设置学习率调度器
        self.deformation_code_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deformation_code_lr_init * self.spatial_lr_scale,
            lr_final=training_args.deformation_code_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.deformation_code_lr_delay_mult,
            max_steps=training_args.deformation_code_lr_max_steps
        )

        # 为 inn_network 设置学习率调度器
        self.inn_network_scheduler_args = get_expon_lr_func(
            lr_init=training_args.inn_lr_init * self.spatial_lr_scale,
            lr_final=training_args.inn_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.inn_lr_delay_mult,
            max_steps=training_args.inn_lr_max_steps
        )
        
    @staticmethod
    def generate_init_tetrahedra(fused_point_cloud, scale):
        # Convert fused_point_cloud to a PyTorch tensor, ensuring data type is float32
        fused_point_cloud = torch.tensor(fused_point_cloud, dtype=torch.float32)
        
        # Define the vertices of a standard regular tetrahedron at the origin, with each vertex distance scaled
        tetrahedron_base = torch.tensor([
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1]
        ], dtype=torch.float32).to(scale.device) * (scale[:,None,:] / torch.sqrt(torch.tensor(3.0)))

        # Get the number of points
        num_points = fused_point_cloud.shape[0]
        
        # Generate random quaternions for rotation
        rand_quats = torch.randn(num_points, 4)  # Generate 4 random components for each quaternion
        rand_quats = rand_quats / rand_quats.norm(dim=1, keepdim=True)  # Normalize each quaternion

        # Extract quaternion components
        qx, qy, qz, qw = rand_quats[:, 0], rand_quats[:, 1], rand_quats[:, 2], rand_quats[:, 3]

        # Construct rotation matrices from the quaternions
        rotation_matrices = torch.zeros((num_points, 3, 3), dtype=torch.float32)
        rotation_matrices[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
        rotation_matrices[:, 0, 1] = 2 * (qx * qy - qz * qw)
        rotation_matrices[:, 0, 2] = 2 * (qx * qz + qy * qw)
        rotation_matrices[:, 1, 0] = 2 * (qx * qy + qz * qw)
        rotation_matrices[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
        rotation_matrices[:, 1, 2] = 2 * (qy * qz - qx * qw)
        rotation_matrices[:, 2, 0] = 2 * (qx * qz - qy * qw)
        rotation_matrices[:, 2, 1] = 2 * (qy * qz + qx * qw)
        rotation_matrices[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

        # Expand the tetrahedron base vertices to match the number of points
        tetrahedra_expanded = tetrahedron_base.unsqueeze(-1)

        # Apply random rotation to each tetrahedron
        rotated_tetrahedra = torch.bmm(rotation_matrices.unsqueeze(1).to(tetrahedra_expanded.device).expand(-1,4,-1,-1).reshape(-1,3,3), tetrahedra_expanded.reshape(-1,3,1)).reshape(-1,4,3)

        
        # Translate each tetrahedron to its respective position in the fused point cloud
        final_tetrahedra = rotated_tetrahedra + fused_point_cloud.unsqueeze(1)

        return final_tetrahedra  # Shape: (N, 4, 3)
    
    def find_unique_face(self):
        
        
        return 

    def update_learning_rate(self, iteration):
        ''' 学习率调度，每步更新 '''
        # 更新 deformation_code 的学习率
        deformation_code_lr = self.deformation_code_scheduler_args(iteration)
        # 更新 inn_network 的学习率
        inn_network_lr = self.inn_network_scheduler_args(iteration)
        # 如果还有其他需要更新的学习率，可以在这里添加
        xyz_network_lr = self.xyz_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deformation_code":
                param_group['lr'] = deformation_code_lr
            elif param_group["name"] == "inn_network":
                param_group['lr'] = inn_network_lr
            elif param_group["name"] == "xyz":
                param_group['lr'] = xyz_network_lr
            # 其他参数的学习率更新
            
    def get_xyz_lr(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = param_group['lr'] 
                return lr
        

    def construct_list_of_attributes(self):
        raise NotImplementedError("not implemented list of attributes function")
        # l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # # All channels except the 3 DC
        # for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        #     l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        # l.append('opacity')
        # for i in range(self._scaling.shape[1]):
        #     l.append('scale_{}'.format(i))
        # for i in range(self._rotation.shape[1]):
        #     l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        raise NotImplementedError("not implemented save ply function")
    
    # def reset_density(self):
    #     if self.use_opacity:
    #         with torch.no_grad():
    #             opacity_new = self.inverse_opacity_activation(0.1*torch.ones((tetrahedra_vertices.shape[0],1),device=tetrahedra_vertices.device,dtype=tetrahedra_vertices.dtype))
    #         optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
    #         self._density = optimizable_tensors["density"] 

    #     else:    
    #         with torch.no_grad():
    #             tet_scale = self.get_scale(self.get_tetra_v)
    #             # computer average scale of each vertices
    #             vertex_indices = self._cells.view(-1).long()  # [T*4]
    #             scales_repeated = tet_scale.view(-1).repeat_interleave(4)  # [T*4]
    #             vertex_scale_sum = torch.zeros(self.num_vertices, device=self._xyz.device).scatter_add_(0, vertex_indices, scales_repeated) # [N]
    #             vertex_scale_count = torch.zeros(self.num_vertices, device=self._xyz.device).scatter_add_(0, vertex_indices, torch.ones_like(scales_repeated)) # [N]
    #             average_scale = vertex_scale_sum / vertex_scale_count.clamp(min=1) # [N]

    #             densities_new = self.inverse_density_activation(alpha_to_density(torch.ones_like(average_scale)[...,None]*0.1,average_scale[...,None]))
    #         optimizable_tensors = self.replace_tensor_to_optimizer(densities_new, "density")
    #         self._density = optimizable_tensors["density"]
        

    def reset_opacity(self):
      
        opacities_new = self.inverse_cell_opacity(torch.min(self.get_cell_opacity, torch.ones_like(self.get_cell_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "cell_opacity")
        self._cell_o = optimizable_tensors["cell_opacity"]

    def load_ply(self, path):
        raise NotImplementedError("not implemented load ply function")

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "cell_opacity" or group["name"] == "scale":
                stored_state = self.optimizer.state.get(group['params'][0], None) 
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def valid_ponits(self):
        """
            return valid points mask (points belong to tetrahedra are valid)
        """
        vertex_indices = self._cells.view(-1)  # [T*4]
        num_vertices = self._xyz.shape[0]
        vertex_scale_count = torch.zeros(num_vertices, device=self._xyz.device, dtype=torch.long).scatter_add_(0, vertex_indices.long(), torch.ones_like(vertex_indices,dtype=torch.long))
        valid_mask = vertex_scale_count > 0
        print(f"prune {(~valid_mask).sum()}   points")
        self.prune_points(~valid_mask)
        # 获取有效顶点的索引
        valid_indices = torch.nonzero(valid_mask).view(-1)  # [num_valid_vertices]
        
        # 创建从原顶点索引到有效顶点索引的映射
        # 有效顶点的旧索引映射到新的连续索引
        index_map = torch.full((self.num_vertices,), -1, dtype=torch.long, device=self._xyz.device)
        index_map[valid_indices] = torch.arange(len(valid_indices), device=self._xyz.device)
        
        # 使用新的索引映射更新 _cells
        # 通过将每个顶点的索引映射到新的有效顶点索引
        new_cells = index_map[self._cells.view(-1)].view(self._cells.shape)
        
        # 将更新后的 new_cells 赋值回 self._cells
        self._cells = new_cells
        
        return valid_mask
    
    
    def _prune_optimizer_points(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "f_dc" or group["name"] == "f_rest" or group["name"] == "density" :
                stored_state = self.optimizer.state.get(group['params'][0], None) 
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        self._xyz = self._xyz[valid_points_mask]
        optimizable_tensors = self._prune_optimizer_points(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._density = optimizable_tensors["density"]
        self._opacity = optimizable_tensors["density"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        # self._rotation = optimizable_tensors["rotation"]

    
    def prune_tetras(self, mask):
        valid_points_mask = ~mask
        self._cells = self._cells[valid_points_mask]
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._density = optimizable_tensors["density"]
        self._scale = optimizable_tensors["scale"]
        self._cell_o = optimizable_tensors["cell_opacity"]
        # self._rotation = optimizable_tensors["rotation"]

        self.cell_gradient_accum = self.cell_gradient_accum[valid_points_mask]
        self.cell_gradient_accum_abs = self.cell_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != "deformation_code" and group["name"] != "inn_network": 
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacity, new_cell_o, new_scale, new_cells, deleted_cells):
        d = {
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "cell_opacity": new_cell_o,
        "scale":new_scale,
        "density":new_opacity,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._scale = optimizable_tensors["scale"]
        self._cell_o = optimizable_tensors["cell_opacity"]
        self._opacity = optimizable_tensors["density"]

        self._xyz = torch.cat([self._xyz,new_xyz],dim=0)
        self._cells = torch.cat([self._cells,new_cells],dim=0)
        deleted_cells = torch.cat([deleted_cells,torch.zeros(new_cells.shape[0], dtype=deleted_cells.dtype, device=deleted_cells.device)])
        
        
        self.denom = torch.zeros((self.get_cells.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_cells.shape[0]), device="cuda")
        self.cell_gradient_accum = torch.zeros((self.get_cells.shape[0], 1), device="cuda")
        self.cell_gradient_accum_abs = torch.zeros((self.get_cells.shape[0], 1), device="cuda")
        
        
        self.prune_tetras(deleted_cells)
        


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=8):
        n_init_tets = self.get_cells.shape[0]

        
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_tets), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        mean,cov,opacity,weight = self.convert_gaussian()
        was = weight / weight.sum(dim=1, keepdim=True)
        # selected_pts_mask2 = torch.logical_and(torch.where(was.min(dim=1).values < 0.0005, True, False),
        #                                        opacity > 0.5).squeeze(-1)
        # selected_pts_mask = torch.logical_or(selected_pts_mask1, selected_pts_mask2)
        subdivide_cells = self.get_cells[selected_pts_mask]
        subdivide_weights = torch.ones_like(was[selected_pts_mask]) * 0.25
     
        
        if N == 8:
            edge_offsets = torch.tensor([
                [0,1], [0,2], [0,3],
                [1,2], [1,3],
                [2,3]
            ], dtype=torch.long, device=self._cells.device)  # [6,2]
            
            edges = subdivide_cells[:, edge_offsets]
            # unique edge
            edges_sorted, _ = torch.sort(edges, dim=-1)  # [M,6,2]
            edges_reshaped = edges_sorted.reshape(-1, 2)  # [M*6,2]
            unique_edges, inverse_indices = torch.unique(edges_reshaped, dim=0, sorted=True, return_inverse=True)  # unique_edges: [K,2]

            # mid points of edge
            v1 = self.get_xyz()[unique_edges[:,0]]  # [K,3]
            v2 = self.get_xyz()[unique_edges[:,1]]  # [K,3]
            new_xyz = (v1 + v2) / 2.0  # [K,3]
            
            new_cell_o = self._cell_o[selected_pts_mask].repeat(N,1)
            new_scale = self._scale[selected_pts_mask].repeat(N,1)
            opacity = self.opacity_activation(self._opacity)
            new_opacity = self.inverse_opacity_activation((opacity[unique_edges[:,0]] + opacity[unique_edges[:,1]]) / 2.0 )
            # if self.use_opacity:
            #     opacity = self.get_weights()
            #     new_density = self.inverse_opacity_activation((opacity[unique_edges[:,0]] + opacity[unique_edges[:,1]]) / 2.0 )
            # else:
            #     new_density = self.inverse_density_activation((self.get_density[unique_edges[:,0]] + self.get_density[unique_edges[:,1]]) / 2.0)  # [K,1]
            new_features_dc = (self._features_dc[unique_edges[:,0]] + self._features_dc[unique_edges[:,1]]) / 2.0  # [K,3,1]
            new_features_rest = (self._features_rest[unique_edges[:,0]] + self._features_rest[unique_edges[:,1]]) / 2.0  # [K,3,8] 
            
            # indices of new vertices
            num_existing_vertices = self.num_vertices
            new_vertex_indices = torch.arange(num_existing_vertices, num_existing_vertices + unique_edges.shape[0], device=self._cells.device, dtype=self._cells.dtype)  # [K,]

            mid_indices = new_vertex_indices[inverse_indices].reshape(-1, 6)  # [M,6]
            
            # original points
            v0 = subdivide_cells[:,0].unsqueeze(1)  # [M,1]
            v1_orig = subdivide_cells[:,1].unsqueeze(1)  # [M,1]
            v2_orig = subdivide_cells[:,2].unsqueeze(1)  # [M,1]
            v3_orig = subdivide_cells[:,3].unsqueeze(1)  # [M,1]

            #  mid points
            m01 = mid_indices[:,0].unsqueeze(1)  # [M,1]
            m02 = mid_indices[:,1].unsqueeze(1)  # [M,1]
            m03 = mid_indices[:,2].unsqueeze(1)  # [M,1]
            m12 = mid_indices[:,3].unsqueeze(1)  # [M,1]
            m13 = mid_indices[:,4].unsqueeze(1)  # [M,1]
            m23 = mid_indices[:,5].unsqueeze(1)  # [M,1]
            
            new_cells = torch.cat([
        torch.cat([v0, m01, m02, m03], dim=1),  # [M,4]
        torch.cat([m01, v1_orig, m12, m13], dim=1),  # [M,4]
        torch.cat([m02, m12, v2_orig, m23], dim=1),  # [M,4]
        torch.cat([m03, m13, m23, v3_orig], dim=1),  # [M,4]
        torch.cat([m01, m02, m03, m12], dim=1),  # [M,4]
        torch.cat([m01, m12, m03, m13], dim=1),  # [M,4]
        torch.cat([m02, m12, m23, m03], dim=1),  # [M,4]
        torch.cat([m03, m13, m12, m23], dim=1),  # [M,4]
    ], dim=0)  # [M*8,4]
            
            deleted_cells = selected_pts_mask
            
        elif N == 4:
               
            # 获取原始顶点索引
            v0 = subdivide_cells[:, 0]  # [M]
            v1 = subdivide_cells[:, 1]
            v2 = subdivide_cells[:, 2]
            v3 = subdivide_cells[:, 3]

            # 获取原始顶点坐标
            p0 = self.get_xyz()[v0]  # [M, 3]
            p1 = self.get_xyz()[v1]
            p2 = self.get_xyz()[v2]
            p3 = self.get_xyz()[v3]

            num_existing_vertices = self.num_vertices
            vertices = torch.stack([p0, p1, p2, p3], dim=1)  # [M, 4, 3]
            new_xyz = torch.sum(subdivide_weights * vertices, dim=1)  # [M, 3]    
            new_vertex_indices = torch.arange(
            num_existing_vertices,
            num_existing_vertices + new_xyz.shape[0],
            device=self._cells.device,
            dtype=self._cells.dtype
            )  # [M]

            new_cells = torch.cat([
                torch.stack([v0, v1, v2, new_vertex_indices], dim=1),  # [M, 4]
                torch.stack([v0, v1, v3, new_vertex_indices], dim=1),
                torch.stack([v0, v2, v3, new_vertex_indices], dim=1),
                torch.stack([v1, v2, v3, new_vertex_indices], dim=1),
            ], dim=0)  # [4*M, 4]
            
            if self.use_opacity:
                new_density = self.inverse_opacity_activation(torch.sum(subdivide_weights * self.get_weights()[subdivide_cells], dim=-2) )
            else:
                new_density = torch.sum(subdivide_weights * self._density[subdivide_cells], dim=-2)  # [M]
            
            new_features_dc = torch.sum(subdivide_weights[..., None] * self._features_dc[subdivide_cells], dim=1)  # [M, 3, 1]
            new_features_rest = torch.sum(subdivide_weights[..., None] * self._features_rest[subdivide_cells], dim=1)  # [M, 3, 8]
            
            deleted_cells = selected_pts_mask
        else:
            raise NotImplementedError("not implemented split function")

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_cell_o, new_scale, new_cells, deleted_cells)

        # prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        # self.prune_points(prune_filter)
        

    def volume_preserving_loss(self):
        vertices = self.get_tetra_v
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]
        d = vertices[:, 3]
        return torch.abs(torch.det(torch.stack([b-a, c-a, d-a], dim=1)) - self._init_volume.detach()).sum()
    
    def init_volume_compute(self):
        vertices = self.get_tetra_v
        a = vertices[:, 0]
        b = vertices[:, 1]
        c = vertices[:, 2]
        d = vertices[:, 3]
        self._init_volume = torch.det(torch.stack([b-a, c-a, d-a], dim=1))
        
    
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        modifier_step = 1.5
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        mean,cov,opacity,weight = self.convert_gaussian()
        cov3d = inverse_strip(cov)
        gs_scales = torch.sqrt(torch.svd(cov3d)[1][...,0])
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                            gs_scales <= self.percent_dense*scene_extent)

        # self.boundary
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scale[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.cell_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.cell_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        
        # self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads_abs, max_grad, extent)
  
        # prune_mask = (self.get_cell_opacity < min_opacity).squeeze()
        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # self.prune_tetras(prune_mask)
        # print(f"delete prune_mask.sum() tetras")
        # self.update_signed_volume()
        torch.cuda.empty_cache()
    
    # check abs grad
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.cell_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.cell_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def get_edge_lengths(self):
        """计算每个四面体的所有边长"""
        vertices = self.get_tetra_v  # [T, 4, 3]
        
        # 计算所有6条边的长度
        edges = torch.stack([
            vertices[:, 0] - vertices[:, 1],  # edge 0-1
            vertices[:, 0] - vertices[:, 2],  # edge 0-2
            vertices[:, 0] - vertices[:, 3],  # edge 0-3
            vertices[:, 1] - vertices[:, 2],  # edge 1-2
            vertices[:, 1] - vertices[:, 3],  # edge 1-3
            vertices[:, 2] - vertices[:, 3],  # edge 2-3
        ], dim=1)  # [T, 6, 3]
        
        # 计算每条边的长度
        edge_lengths = torch.norm(edges, dim=-1)  # [T, 6]
        return edge_lengths

    def edge_ratio_loss(self, epsilon=1e-6,r=2.0):
        """
        计算每个四面体的最长边与最短边的比例loss
        loss = max/min - 1，当所有边长相等时loss为0
        
        Args:
            epsilon: 防止除零的小值
            
        Returns:
            loss: 边长比例loss
        """
        edge_lengths = self.get_edge_lengths()  # [T, 6]
        
        # 获取每个四面体的最长边和最短边
        max_edges = torch.max(edge_lengths, dim=-1)[0]  # [T]
        min_edges = torch.min(edge_lengths, dim=-1)[0]  # [T]
        
        # 计算比例（添加epsilon防止除零）并减1
        ratios = torch.max(max_edges / (min_edges + epsilon) - r, torch.zeros_like(max_edges))
        
        # 计算平均loss
        loss = torch.mean(ratios)
        
        return loss

if __name__ == "__main__":
    tets = TetrahedraModel(sh_degree=3)