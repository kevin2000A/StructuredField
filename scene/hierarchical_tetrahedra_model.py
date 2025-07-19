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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.sh_utils import eval_sh
import math 
from utils.geo_utils import compute_rms_edge_ratio, compute_tetrahedron_signed_volume

# rgb = render_htet(viewpoint_camera,self,torch.tensor([1.0,1.0,1.0],device=self._xyz.device),lod=0,cell_mask=selected_pts_mask)['render'].clamp(0.0,1.0)
# torchvision.utils.save_image(rgb, "/home/juyonggroup/kevin2000/Desktop/gaussian-splatting/scene/debug0.png")
def render_htet(viewpoint_camera, tets, bg_color : torch.Tensor, lod=0, scaling_modifier = 1.0, override_color = None, cell_mask=None, feature_interp_mode="sh_first"):
    """
    渲染层级四面体模型场景。
    
    Args:
        viewpoint_camera: 视点相机
        tets: HierarchicalTetrahedraModel实例
        pipe: 渲染管线
        bg_color: 背景颜色张量（必须在GPU上）
        lod: 使用的LOD层级，默认为0（最粗糙层级）
        scaling_modifier: 缩放修改器
        override_color: 可选的覆盖颜色
        feature_interp_mode: 特征插值模式，可选值：
            - "sh_first": 先插值SH系数再转RGB（默认）
            - "rgb_first": 先转RGB再插值
    
    Returns:
        渲染结果字典
    """
 
    # 获取当前LOD层级的单元数
    cells = tets.get_cells(lod)
    
    # 创建零张量，用于梯度回传（screen-space means）
    screenspace_points = torch.zeros((cells.shape[0], 4), dtype=tets._xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=tets.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取单元中心，协方差和不透明度
    means3D, cov3D_precomp = tets.get_gs_mean_covs(lod)
    opacities = tets.get_opacities(lod)
    
    # 获取顶点权重（用于插值）
    weights = tets.weight_activation(tets._weights)[cells]  # [T, 4, 1]
    
    # 特征处理
    if feature_interp_mode == "sh_first":
        # 路径1: 先插值SH系数再转RGB
        features = tets.get_gs_features(lod)  # 获取已经插值后的单元SH系数
        
        # 将特征转换为可用于SH评估的格式
        features_view = features.view(-1, 3, (tets.max_sh_degree+1)**2)
        
        # 计算视角方向
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        
        # 评估SH并转换为RGB
        sh2rgb = eval_sh(tets.active_sh_degree, features_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
    else:  # "rgb_first"
        # 路径2: 先转RGB再插值
        # 获取顶点特征和坐标
        vertex_features = tets.get_vertices_features()  # 获取顶点SH系数
        vertices = tets.get_vertices(lod)
        
        # 计算视角方向（对每个顶点）
        dir_pp = (vertices - viewpoint_camera.camera_center.repeat(vertices.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        
        # 将顶点特征转换为可用于SH评估的格式
        features_view = vertex_features.view(-1, 3, (tets.max_sh_degree+1)**2)
        
        # 评估SH并转换为RGB
        vertex_colors = eval_sh(tets.active_sh_degree, features_view, dir_pp_normalized)
        vertex_colors = torch.clamp_min(vertex_colors + 0.5, 0.0)
        
        # 获取单元顶点的颜色
        cell_vertex_colors = vertex_colors[cells]  # [T, 4, 3]
        
        # 使用权重插值顶点颜色得到单元颜色
        weights_expanded = weights / torch.sum(weights, dim=1, keepdim=True)  # [T, 4, 1]
        colors_precomp = torch.sum(cell_vertex_colors * weights_expanded, dim=1)  # [T, 3]
    
    # 光栅化可见的高斯体到图像，获取它们的屏幕半径
    if cell_mask is not None:
        means3D = means3D[cell_mask]
        cov3D_precomp = cov3D_precomp[cell_mask]
        opacities = opacities[cell_mask]
        colors_precomp = colors_precomp[cell_mask]
        
    rendered_image, radii, inv_depth, mask = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D_precomp)

    # 返回渲染结果
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "render_alpha": mask,
        "radii": radii
    }

def inverse_softplus(y, beta=1.0, min_value=1e-8):
    # 确保y值大于最小阈值，因为softplus的值域是(0,∞)
    y_safe = torch.clamp(y, min=min_value)
    # 使用expm1(y)计算e^y-1，数值上比exp(y)-1更稳定
    return torch.log(torch.expm1(beta * y_safe)) / beta

class HierarchicalTetrahedraModel:
    
    def setup_functions(self):
        
        def build_mean_covariance_from_weighted_vertices(vertices, weight, scaling_modifier = 1.0, cov_modifier = 1e-9, rotation_quaternion=None):
            """

            Args:
                vertices (_torch.tensor_): [T,4,3]
                weight (_torch.tensor_): [T,4,1]
                scaling_modifier (float, optional): _description_. Defaults to 1.0.
                rotation_quaternion (_torch.tensor_, optional): [T,4] quaternion for rotation. Defaults to None.

            Returns:
                _type_: _description_
            """
            assert vertices.shape[1] == 4, "Vertices must be of shape (N, 4, 3)"
            assert weight.shape[1] == 4, "Weights must be of shape (N, 4, 1)"
            assert weight.shape[2] == 1, "Weights must be of shape (N, 4, 1)"
            
            sum_weights = torch.sum(weight, dim=1).clamp(min=1e-8) # 添加 clamp 以防止除以零
            mean = torch.sum(vertices * weight, dim=1) / sum_weights
            
            diff = (vertices - mean[:,None,:]).unsqueeze(-1)
            
            # 也对协方差计算中的分母进行 clamp
            cov = torch.sum(weight[...,None] * diff @ diff.transpose(-1, -2), dim=1) / (sum_weights)[...,None] * scaling_modifier + cov_modifier * torch.eye(3, device=vertices.device)
            
            # Apply rotation if provided
            if rotation_quaternion is not None:
                R = build_rotation(rotation_quaternion) # [T, 3, 3]
                cov = R @ cov @ R.transpose(-1, -2) # R * Cov * R^T

            # / (sum_weights)[...,None]
            # cov = torch.sum(diff @ diff.transpose(-1, -2), dim=1)  * scaling_modifier + cov_modifier * torch.eye(3, device=vertices.device)
            
            return mean, strip_symmetric(cov)


        self.density_activation = torch.relu
        self.inverse_density_activation = torch.relu

        # 顶点权重激活函数 - 根据weight_activation_type选择不同的激活函数
        if self.weight_activation_type == "relu":
            self.weight_activation = torch.relu  # 使用ReLU确保权重为正值
            self.inverse_weight_activation = torch.relu  # 反向激活函数也是ReLU
        else:  # "exp"为默认值
            self.weight_activation = torch.exp  # 使用exp确保权重为正值
            self.inverse_weight_activation = torch.log  # 反向激活函数

        # 顶点权重激活函数
        self.opacity_activation = torch.sigmoid 
        self.inverse_opacity_activation = inverse_sigmoid  
       

        self.mean_covariance_activation = build_mean_covariance_from_weighted_vertices
        self.scale_activation = lambda x: torch.nn.functional.softplus(x, beta=1, threshold=10)
        self.inverse_scale_activation = lambda x: torch.where(x > 10, x, inverse_softplus(x, beta=1, min_value=1e-8))
        
        self.rotation_activation = torch.nn.functional.normalize

        # 现在只存储前三个坐标，第四个坐标由1-前三个之和计算
        self.bary_coords_activation = lambda x: x
        self.inverse_bary_coords_activation = lambda x: x
        
        # self.bary_coords_activation = torch.relu
        # self.inverse_bary_coords_activation = torch.relu

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


    def __init__(self, sh_degree, scaling_modifier=2.0, use_cell_opacity=True, use_cell_scale=False, max_depth=4, optimizer_type="default", weight_activation_type="exp", optimizable_rotation=False):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        # 设置权重激活函数类型，可选："relu"或"exp"
        self.weight_activation_type = weight_activation_type
        self.optimizable_rotation = optimizable_rotation
        
        # Root vertices coordinates (equivalent to root_vertices_coords in plan)
        self._xyz = torch.empty(0)
        
        # 添加顶点权重
        self._weights = torch.empty(0)
        
        # 设置层级结构变量
        self.setup_hierarchy_variables(max_depth)
        
        # Attributes for features and appearance
        self._features_dc = torch.empty(0)       # DC component of spherical harmonics
        self._features_rest = torch.empty(0)     # Rest components of spherical harmonics
        self._cell_scale = torch.empty(0)        # Scale for controlling Gaussian size, [T, 1]
        self._cell_rotation = torch.empty(0)     # Rotation for controlling Gaussian orientation (quaternion), [T_finest, 4]
        
        # 控制是否使用单元级缩放参数
        self.use_cell_scale = use_cell_scale
        
        # Opacity related attributes
        self.use_cell_opacity = use_cell_opacity
        if use_cell_opacity:
            self._cell_o = torch.empty(0)        # Per-cell opacity
        else:
            self._opacity = torch.empty(0)       # Per-vertex opacity
            
        # Attributes for tracking rendering and training statistics
        self.max_radii2D = torch.empty(0)        # For finest LOD cells, [T_max_depth, 1]
        self.cell_gradient_accum = torch.empty(0) # For finest LOD cells, [T_max_depth, 1]
        self.cell_gradient_accum_abs = torch.empty(0) # For finest LOD cells, [T_max_depth, 1]
        self.denom = torch.empty(0)  # For finest LOD cells, [T_max_depth, 1]
        self._init_volume = torch.empty(0)  # For coarset LOD cells, [T_depth_0, 1]

        
        # Optimizer and training related
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        # Other parameters
        self.scaling_modifier = scaling_modifier
        self.alpha_ratio = 1.0
        
        # INN network (deformation)
        self.inn_network = None
        self.deformation_code = None
        self.freeze = False
        
        # Gradient clipping for INN
        self.clip_grad_norm_inn = True
        self.max_grad_norm_inn = 1.0
        self.render_tet_mask = None
        # Setup activation functions
        self.setup_functions()

    def setup_hierarchy_variables(self, absolute_max_lineage_depth=4):
        """
        初始化层级结构的变量
        
        约定：
        - 根节点深度为0，最大深度为d，总共有d+1个深度层级
        - level_bary_coords 和 level_parent_vertex_indices 有d个元素（索引0到d-1）
        - level_cells 和 level_cell_lineage_depths 有d+1个元素（索引0到d）
        - level_cell_lineage_depths[0] 初始化为全0（根单元的细分深度为0）
        """
        # Hierarchical structure data
        self.level_bary_coords = nn.ParameterList([])  # 使用ParameterList存储可优化的重心坐标 [P_d, 3]
                                                      # 第四个坐标由1-前三个之和得到
        self.level_parent_vertex_indices = []    # List of Tensors [P_d, 4] (static topology)
        
        # 修改：直接让level_cells存储四面体拓扑，而不是索引
        self.level_cells = []                    # List of Tensors [T_d, 4] (存储每个LOD层级的四面体顶点索引)
        self.level_cell_lineage_depths = []      # List of Tensors [T_d] (static lineage depth)
        
        # Hierarchical state variables
        self.num_root_vertices = 0               # Will be set in create_from_tetra
        self.global_vertex_count = 0             # Total vertices: num_root_vertices + virtual points
        self.current_lod_depth = 0               # 当前最深层级的LOD索引(比如有5个LOD，lod0,lod1,lod2,lod3,lod4, 那么current_lod_depth = 4)
        self.absolute_max_lineage_depth = absolute_max_lineage_depth      # Default max depth for subdivision lineage

    def create_from_tetra(self, tetra_data, spatial_lr_scale: float, low_scale_init: bool, training_args=None):
        """
        从 BasicTetrahedra 数据初始化模型的根网格层级。
        
        Args:
            tetra_data: BasicTetrahedra 类型数据，包含顶点和四面体单元
            spatial_lr_scale: 空间坐标学习率的缩放因子
            training_args: 可选的训练参数，用于设置优化器
        """
        self.spatial_lr_scale = spatial_lr_scale
        

        
        # 1. 设置根顶点坐标
        fused_xyz = torch.tensor(np.asarray(tetra_data.vertices), dtype=torch.float32).cuda()
        self._xyz = nn.Parameter(fused_xyz.contiguous())  # 注册为可优化参数
        self.num_root_vertices = fused_xyz.shape[0]
        self.global_vertex_count = self.num_root_vertices
        
        # 初始化顶点权重为1.0 并设为可优化量
        init_weights = torch.ones((self.num_root_vertices, 1), dtype=torch.float32, device=fused_xyz.device) * 0.25
        self._weights = nn.Parameter(self.inverse_weight_activation(init_weights).contiguous())  # 注册为可优化参数
        
        # 2. 设置初始四面体单元（LOD 0，即深度0的根单元）
        initial_cells = torch.tensor(np.asarray(tetra_data.cells), dtype=torch.long).cuda()
        
        # 根据层级深度约定，添加深度0的单元 - 直接存储拓扑
        self.level_cells = [initial_cells]
        
        # 初始化深度0的单元谱系深度为全0
        self.level_cell_lineage_depths = [
            torch.zeros(initial_cells.shape[0], dtype=torch.long, device=initial_cells.device)
        ]
        
        # 设置当前LOD深度为0（表示最深层级的索引为0，即只有一个层级LOD 0）
        self.current_lod_depth = 0
        
        # 3. 初始化特征 (SH coefficients)
        # 为根顶点创建特征
        if tetra_data.colors is not None:
            fused_color_sh = RGB2SH(torch.tensor(np.asarray(tetra_data.colors), dtype=torch.float32).cuda())
        else:
            # 如果没有提供颜色，使用随机颜色
            fused_color_sh = RGB2SH(torch.rand((self.num_root_vertices, 3), device=fused_xyz.device))

        
        # 创建特征张量
        
        features = torch.zeros((fused_xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :, 0] = fused_color_sh  # 设置DC分量
        features[:, 3:, 1:] = 0.0
        
        # 分割为DC和rest部分，并注册为可优化参数
        self._features_dc = nn.Parameter(features[:, :, 0:1].clone().contiguous())
        self._features_rest = nn.Parameter(features[:, :, 1:].clone().contiguous())
        
        # 4. 初始化四面体相关属性
        num_initial_cells = initial_cells.shape[0]
        
        # 计算并存储每个四面体的初始体积（可用于体积保持正则化）
        tetra_vertices = fused_xyz[initial_cells]  # [T, 4, 3]
        tetra_scale = self.get_scale(tetra_vertices)  # [T, 1] 或相似形状
        if low_scale_init:
            tetra_scale = torch.ones_like(tetra_scale) * 0.1
        # 5. 初始化透明度和缩放参数
        # 根据use_cell_scale决定如何初始化缩放参数
        if self.use_cell_scale:
            # 使用单元级缩放参数，直接初始化为四面体体积的立方根
            self._cell_scale = nn.Parameter(self.inverse_scale(tetra_scale).contiguous())  # [T, 1]

        
        # 根据 use_cell_opacity 初始化透明度
        if self.use_cell_opacity:
            # 使用单元级透明度
            init_cell_opacity = self.inverse_cell_opacity(torch.ones_like(tetra_scale) * 0.1)  # 初始透明度为0.1
            self._cell_o = nn.Parameter(init_cell_opacity.contiguous())  # 注册为可优化参数
            self._opacity = torch.empty(0,device=fused_xyz.device)  # 不使用顶点透明度
        else:
            # 使用顶点级透明度
            init_opacity = self.inverse_opacity_activation(torch.ones((self.num_root_vertices, 1), device=fused_xyz.device) * 0.1)  # 初始透明度为0.1
            self._opacity = nn.Parameter(init_opacity.contiguous())  # 注册为可优化参数
            self._cell_o = torch.empty(0,device=fused_xyz.device)  # 不使用单元透明度

        # 初始化旋转四元数（如果启用了优化旋转）
        if self.optimizable_rotation:
            # 初始化为单位四元数 [w,x,y,z] = [1,0,0,0]，表示无旋转
            init_rotation = torch.zeros((num_initial_cells, 4), device=fused_xyz.device)
            init_rotation[:, 0] = 1.0  # w分量设为1，表示单位四元数
            # 归一化确保是单位四元数
            init_rotation = self.rotation_activation(init_rotation)
            self._cell_rotation = nn.Parameter(init_rotation.contiguous())
        else:
            self._cell_rotation = torch.empty(0, device=fused_xyz.device)

        # 6. 初始化其他训练和渲染相关参数
        self.max_radii2D = torch.zeros(num_initial_cells, device=fused_xyz.device)
        self.cell_gradient_accum = torch.zeros((num_initial_cells, 1), device=fused_xyz.device)
        self.cell_gradient_accum_abs = torch.zeros((num_initial_cells, 1), device=fused_xyz.device)
        self.denom = torch.zeros((num_initial_cells, 1), device=fused_xyz.device)
        
        # 存储初始体积用于可能的正则化
        v0 = tetra_vertices[:, 0]
        v1 = tetra_vertices[:, 1]
        v2 = tetra_vertices[:, 2]
        v3 = tetra_vertices[:, 3]
        self._init_volume = torch.abs(torch.det(torch.stack([v1-v0, v2-v0, v3-v0], dim=1)))

        init_render_tet_mask = torch.ones((num_initial_cells, 1), device=fused_xyz.device, dtype=torch.bool)
        self.render_tet_mask = init_render_tet_mask
     
        # 9. 确保四面体具有正体积
        self.update_positive_signed_volume()



    def get_scale(self, vertices_for_tetrahedra):
        """
        计算四面体的缩放因子，基于体积的立方根。
        
        Args:
            vertices_for_tetrahedra: 形状为 [T, 4, 3] 的张量，表示T个四面体每个的4个顶点
            
        Returns:
            torch.Tensor: 形状为 [T, 1] 的张量，表示每个四面体的缩放因子
        """
        a = vertices_for_tetrahedra[:, 0]  # [T, 3]
        b = vertices_for_tetrahedra[:, 1]
        c = vertices_for_tetrahedra[:, 2]
        d = vertices_for_tetrahedra[:, 3]
        
        # 计算四面体体积并取立方根作为缩放因子
        volume = torch.abs(torch.det(torch.stack([b-a, c-a, d-a], dim=1)))  # [T]
        scale = volume ** (1/3)  # [T]
        
        return scale.unsqueeze(-1)  # [T, 1]
    
    def update_positive_signed_volume(self):
        """
        确保所有四面体具有正体积，如果发现负体积则交换顶点顺序。
        
        Args:
            xyz: 顶点坐标张量
        """
        if not self.level_cells:
            return
        
        xyz = self._xyz
        # 获取根层级的单元拓扑
        cells = self.level_cells[0]
        
        v0 = xyz[cells[:, 0]]
        v1 = xyz[cells[:, 1]]
        v2 = xyz[cells[:, 2]]
        v3 = xyz[cells[:, 3]]
        
        v01 = v1 - v0
        v02 = v2 - v0
        v03 = v3 - v0
        
        # 计算每个四面体的有向体积
        volume = torch.det(torch.stack([v01, v02, v03], dim=-1))
        
        # 找出体积为负的四面体
        negative_volume_mask = volume < 0
        
        # 交换负体积四面体的顶点顺序 (交换v2和v3)
        if negative_volume_mask.any():
            # 直接修改 level_cells[0] 中的顶点顺序
            cells_to_fix = cells[negative_volume_mask]
            
            temp = cells_to_fix[:, 2].clone()
            cells_to_fix[:, 2] = cells_to_fix[:, 3]
            cells_to_fix[:, 3] = temp
            
            self.level_cells[0][negative_volume_mask] = cells_to_fix

    def training_setup(self, training_args):
        """设置优化器和学习率调度器"""
        # 保存训练参数
        self.percent_dense = training_args.percent_dense
        
        # 添加梯度裁剪参数
        self.clip_grad_norm_inn = getattr(training_args, 'clip_grad_norm_inn', False)
        self.max_grad_norm_inn = getattr(training_args, 'max_grad_norm_inn', 1.0)
        
        # 初始化统计数据 - 使用最深层级的单元数量
        if len(self.level_cells) > 0:
            deepest_cells = self.level_cells[-1]
            num_deepest_cells = deepest_cells.shape[0]
            self.cell_gradient_accum = torch.zeros((num_deepest_cells, 1), device="cuda")
            self.cell_gradient_accum_abs = torch.zeros((num_deepest_cells, 1), device="cuda")
            self.denom = torch.zeros((num_deepest_cells, 1), device="cuda")
        
        # 初始化优化器参数列表
        params = [
            # 根顶点坐标
           #  {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "_xyz"},
            # 顶点权重
            {'params': [self._weights], 'lr': training_args.weight_lr, "name": "weights"},
            # 特征参数 
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
        ]
        self.bary_lr = training_args.bary_lr_init
        # 添加INN网络和变形码参数
        if self.inn_network is not None:
            params.append({'params': self.inn_network.parameters(), 'lr': training_args.inn_lr_init * self.spatial_lr_scale, "name": "inn_network"})
        
        if hasattr(self, 'deformation_code') and self.deformation_code is not None:
            params.append({'params': [self.deformation_code], 'lr': training_args.deformation_code_lr_init * self.spatial_lr_scale, "name": "deformation_code"})
        
        # 添加层级重心坐标参数
        for i, bary_coord in enumerate(self.level_bary_coords):
            params.append({'params': [bary_coord], 'lr': training_args.bary_lr_init, "name": f"level_bary_coords_{i}"})
        
        # 添加透明度相关参数
        if self.use_cell_opacity:
            params.append({'params': [self._cell_o], 'lr': training_args.cell_opacity_lr, "name": "cell_opacity"})
        else:
            params.append({'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"})
        
        # # 添加缩放参数
        # params.append({'params': [self._cell_scale], 'lr': training_args.scale_lr, "name": "scale"})
        
        # 如果使用单元级缩放参数，也添加它
        if self.use_cell_scale:
            params.append({'params': [self._cell_scale], 'lr': training_args.scale_lr, "name": "cell_scale"})
            
        # 如果使用可优化旋转，添加旋转参数
        if self.optimizable_rotation and self._cell_rotation.shape[0] > 0:
            # 使用与缩放参数相同的学习率，或者使用专门的旋转学习率（如果有）
            rotation_lr = getattr(training_args, 'rotation_lr', training_args.scale_lr)
            params.append({'params': [self._cell_rotation], 'lr': rotation_lr, "name": "cell_rotation"})
        
        # 创建优化器
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        
        # 设置学习率调度器
        # 为根顶点坐标设置学习率调度器
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )
        
        # 为INN网络设置学习率调度器
        if self.inn_network is not None:
            self.inn_network_scheduler_args = get_expon_lr_func(
                lr_init=training_args.inn_lr_init * self.spatial_lr_scale,
                lr_final=training_args.inn_lr_final * self.spatial_lr_scale,
                lr_delay_mult=training_args.inn_lr_delay_mult,
                max_steps=training_args.inn_lr_max_steps
            )
        
        # 为变形码设置学习率调度器
        if hasattr(self, 'deformation_code') and self.deformation_code is not None:
            self.deformation_code_scheduler_args = get_expon_lr_func(
                lr_init=training_args.deformation_code_lr_init * self.spatial_lr_scale,
                lr_final=training_args.deformation_code_lr_final * self.spatial_lr_scale,
                lr_delay_mult=training_args.deformation_code_lr_delay_mult,
                max_steps=training_args.deformation_code_lr_max_steps
            )
        
        # 为重心坐标设置学习率调度器
        self.bary_scheduler_args = get_expon_lr_func(
            lr_init=training_args.bary_lr_init,
            lr_final=training_args.bary_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        """更新优化器中各参数的学习率"""
        # 更新根顶点坐标的学习率
        xyz_lr = self.xyz_scheduler_args(iteration)
        
        # 更新INN网络学习率
        if hasattr(self, 'inn_network_scheduler_args'):
            inn_lr = self.inn_network_scheduler_args(iteration)
        
        # 更新变形码学习率
        if hasattr(self, 'deformation_code_scheduler_args'):
            deformation_code_lr = self.deformation_code_scheduler_args(iteration)
        
        # 更新重心坐标学习率
        bary_lr = self.bary_scheduler_args(iteration)
        
        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "_xyz":
                param_group['lr'] = xyz_lr
            elif param_group["name"] == "inn_network" and hasattr(self, 'inn_network_scheduler_args'):
                param_group['lr'] = inn_lr
            elif param_group["name"] == "deformation_code" and hasattr(self, 'deformation_code_scheduler_args'):
                param_group['lr'] = deformation_code_lr
            elif param_group["name"].startswith("level_bary_coords_"):
                param_group['lr'] = bary_lr

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


        
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            
    def get_vertices(self, lod):
        """
        获取指定LOD层级用于渲染的顶点坐标。
        对根顶点使用INN网络进行形变，然后根据重心坐标插值计算虚拟点。
        
        Args:
            lod: 指定的LOD层级
            
        Returns:
            torch.Tensor: 形状为[N, 3]的顶点坐标
        """
        # 检查LOD层级有效性
        assert 0 <= lod <= self.current_lod_depth, f"无效的LOD层级: {lod}，有效范围为0到{self.current_lod_depth}"
        
        # 如果有INN网络并且没有冻结，使用INN进行形变
        if self.inn_network is not None and not self.freeze:
            # 对根顶点进行形变
            deform_out = self.inn_network(self.deformation_code, self._xyz)
            deformed_xyz = deform_out
                
            # 计算形变后的所有顶点坐标
            global_vertices_list = [deformed_xyz]
            
            # 获取所有层级的归一化重心坐标
            all_normalized_barycoords = self.get_barycoords(None)  # 返回[P_d, 4]的完整坐标
            
            # 根据LOD层级处理需要的深度
            max_level = len(self.level_parent_vertex_indices)
            
            # 获取虚拟点坐标，逐层计算
            for d in range(max_level):
                parent_indices_at_d = self.level_parent_vertex_indices[d]  # [P_d, 4]
                barycentric_coords_at_d = all_normalized_barycoords[d]  # [P_d, 4] 归一化后的
                
                # 计算当前已知的所有顶点坐标
                temp_all_computed_vertices = torch.cat(global_vertices_list, dim=0)
                # 获取父顶点坐标（可能包括根顶点和之前层级的虚拟点）
                parent_vertex_coords = temp_all_computed_vertices[parent_indices_at_d]  # [P_d, 4, 3]
                
                # 使用归一化的重心坐标线性插值计算新的虚拟点坐标
                bary_expanded = barycentric_coords_at_d.unsqueeze(-1)  # [P_d, 4, 1]
                new_virtual_vertices_at_d = torch.sum(parent_vertex_coords * bary_expanded, dim=1)  # [P_d, 3]
                
                # 添加到全局顶点列表
                global_vertices_list.append(new_virtual_vertices_at_d)
            
            return torch.cat(global_vertices_list, dim=0)
        else:
            # 不使用形变网络，直接返回原始顶点
            return self.get_all_vertices(depth=lod)

    def get_cells(self, lod):
        """
        获取指定LOD层级的单元拓扑。
        
        Args:
            lod: 指定的LOD层级
            
        Returns:
            torch.Tensor: 形状为[T, 4]的单元拓扑
        """
        # 检查LOD层级有效性
        assert 0 <= lod <= self.current_lod_depth, f"无效的LOD层级: {lod}，有效范围为0到{self.current_lod_depth}"
        
        # 直接返回指定LOD层级的单元拓扑
        return self.level_cells[lod]
    
    def get_gs_mean_covs(self, lod):
        """
        计算指定LOD层级的单元中心坐标和协方差矩阵。
        基于当前拓扑关系、顶点坐标和每个顶点的权重。
        
        Args:
            lod: 指定的LOD层级
            
        Returns:
            tuple: (means, covs) 单元中心坐标和协方差矩阵
                means: 形状为[T, 3]的单元中心坐标
                covs: 形状为[T, 6]的单元协方差矩阵（压缩表示）
        """
        # 获取当前形变后的顶点坐标
        vertices = self.get_vertices(lod)
        
        # 获取当前LOD的单元拓扑
        cells = self.level_cells[lod]
        
        # 获取当前激活的顶点权重
        weights = self.weight_activation(self._weights)
        
        # 计算每个单元的4个顶点坐标和对应权重
        cell_vertices = vertices[cells]  # [T, 4, 3]
        cell_weights = weights[cells]    # [T, 4, 1]
        
        # 确定使用的缩放系数
        if self.use_cell_scale:
            # 使用单元级缩放系数 - 注意: _cell_scale 只对应最精细LOD
            # 判断当前LOD是否是最精细的LOD
            if lod == self.current_lod_depth:  # 当前是最精细LOD
                cell_scaling_modifier = self.scale_activation(self._cell_scale).unsqueeze(-1)  # [T_finest, 1]
            else:
                # 非最精细LOD，使用全局缩放系数
                cell_scaling_modifier = self.scale_activation(self._cell_scale_list[lod]).unsqueeze(-1)
        else:
            # 使用全局缩放系数
            cell_scaling_modifier = self.scaling_modifier
        
        # 获取旋转四元数（如果启用）
        rotation_quaternion = None
        if self.optimizable_rotation and self._cell_rotation.shape[0] > 0:
            if lod == self.current_lod_depth:  # 当前是最精细LOD
                rotation_quaternion = self.rotation_activation(self._cell_rotation)  # [T_finest, 4]
            # 非最精细LOD的旋转处理可以在这里添加，例如使用cell_rotation_list
        
        # 使用build_mean_covariance_from_weighted_vertices函数计算均值和协方差
        mean, covs = self.mean_covariance_activation(cell_vertices, cell_weights, cell_scaling_modifier, cov_modifier=1e-8, rotation_quaternion=rotation_quaternion)
        
        return mean, covs
        
    def get_gs_features(self, lod):
        """
        获取指定LOD层级的特征。
        基于当前拓扑关系、顶点特征和每个顶点的权重。
        
        Args:
            lod: 指定的LOD层级
            
        Returns:
            tuple: (features_dc, features_rest) 单元特征
        """
        # 获取当前LOD的单元拓扑
        cells = self.level_cells[lod]
        
        # 获取所有顶点的特征
        all_features_dc = self._features_dc
        all_features_rest = self._features_rest
        
        # 获取所有顶点的权重
        weights = self.weight_activation(self._weights)
        
        # 计算每个单元的4个顶点特征和对应权重
        cell_features_dc = all_features_dc[cells]    # [T, 4, 3, 1]
        cell_features_rest = all_features_rest[cells] # [T, 4, 3, C-1]
        cell_weights = weights[cells].unsqueeze(-1)  # [T, 4, 1, 1]
        
        # 使用权重计算加权特征
        sum_weights = torch.sum(cell_weights, dim=1).clamp(min=1e-8)  # [T, 1, 1]
        
        weighted_features_dc = torch.sum(cell_features_dc * cell_weights, dim=1) / sum_weights  # [T, 3, 1]
        weighted_features_rest = torch.sum(cell_features_rest * cell_weights, dim=1) / sum_weights  # [T, 3, C-1]
        
        return torch.cat((weighted_features_dc, weighted_features_rest), dim=-1)
    
    def get_opacities(self, lod):
        """
        获取指定LOD层级的不透明度。
        如果use_cell_opacity为True，直接使用单元不透明度；
        否则基于顶点不透明度插值计算。
        
        Args:
            lod: 指定的LOD层级
            
        Returns:
            torch.Tensor: 形状为[T, 1]的不透明度值
        """
        # 获取当前LOD的单元拓扑
        cells = self.level_cells[lod]
        
        # 两种情况处理不透明度
        if self.use_cell_opacity:
            # 使用单元级不透明度 - 注意: _cell_o 只对应最精细LOD
            if lod == self.current_lod_depth:  # 当前是最精细LOD
                return self.cell_opacity_activation(self._cell_o)
            else:
                # 非最精细LOD，使用较低的固定不透明度 TODO
                # 这里可以根据实际需求调整，例如使用降低的不透明度或其他策略
                return torch.ones(cells.shape[0], 1, device=cells.device) * 0.5
        else:
            # 使用顶点级不透明度
            all_opacities = self._opacity
            # 获取所有顶点的权重
            weights = self.weight_activation(self._weights)
            
            # 计算每个单元的4个顶点不透明度和对应权重
            cell_opacities = self.cell_opacity_activation(all_opacities[cells])  # [T, 4, 1]
            cell_weights = weights[cells]          # [T, 4, 1]
            
            # 使用权重计算加权不透明度
            sum_weights = torch.sum(cell_weights, dim=1).clamp(min=1e-8)  # [T, 1]
            weighted_opacities = torch.sum(cell_opacities * cell_weights, dim=1) / sum_weights  # [T, 1]
            
            return weighted_opacities

    def get_vertices_features(self):
        """
        获取指定LOD层级的顶点特征。
        基于当前拓扑关系、顶点特征和每个顶点的权重。

        Args:
            lod: 指定的LOD层级
            
        Returns:
            torch.Tensor: 形状为 [N_total, 3] 的张量，表示所有顶点的特征
        """
        return torch.cat((self._features_dc, self._features_rest), dim=-1)
        

    def save_model(self, path):
        # Save all necessary state: root_vertices_coords, all level_... lists,
        # features, opacities, scales, optimizer state, active_sh_degree etc.
        pass

    def load_model(self, path, training_args): # training_args might be needed to re-setup optimizer/schedulers
        pass
        

    def freeze_inn(self):
        self.freeze = True
        if self.inn_network:
            for param in self.inn_network.parameters():
                param.requires_grad = False
        if self.deformation_code is not None and isinstance(self.deformation_code, nn.Parameter):
             self.deformation_code.requires_grad = False


    def unfreeze_inn(self):
        self.freeze = False
        if self.inn_network:
            for param in self.inn_network.parameters():
                param.requires_grad = True
        if self.deformation_code is not None and isinstance(self.deformation_code, nn.Parameter):
            self.deformation_code.requires_grad = True
            
    def reset_inn(self, training_args):
        """
        重新初始化INN网络并将其添加到优化器中。
        
        Args:
            training_args: 包含训练参数的对象，用于重新设置网络和优化器
        """
        # 保存当前优化器状态
        old_opt_state = None
        if self.optimizer is not None:
            old_opt_state = self.optimizer.state_dict()
        
        # 重新初始化INN网络
        if hasattr(training_args, 'model_type'):
            if training_args.model_type == "NDR":
                self.inn_network = DeformNetwork(d_feature=training_args.embedding_dim,
                                            scale_included=True,
                                            n_blocks=training_args.n_blocks,
                                            d_hidden=training_args.hidden_dim,
                                            n_layers=training_args.n_layers,
                                            skip_in=training_args.skip_in,
                                            multires=training_args.multires,
                                            weight_norm=training_args.weight_norm,).to(self._xyz.device)
            elif training_args.model_type == "NVP":
                self.inn_network = NVPSimplified(n_layers=training_args.n_blocks,
                                                feature_dims=training_args.embedding_dim,
                                                hidden_size=training_args.hidden_size,
                                                proj_dims=training_args.proj_dims,
                                                code_proj_hidden_size=[],
                                                proj_type=training_args.proj_type,
                                                pe_freq=training_args.pe_freq,
                                                normalization=training_args.normalization,
                                                activation=nn.LeakyReLU,
                                                device=self._xyz.device).to(self._xyz.device)
        
        # 重新初始化变形码
        if hasattr(training_args, 'embedding_dim'):
            self.deformation_code = nn.Parameter(torch.zeros(training_args.embedding_dim, device=self._xyz.device))
        
        # 确保参数可以参与优化
        self.unfreeze_inn()
        
        # 重新设置优化器
        if old_opt_state is not None:
            # 保存之前的参数组
            old_param_groups = self.optimizer.param_groups
            
            # 找到不包含INN网络和变形码的参数组
            non_inn_param_groups = [pg for pg in old_param_groups 
                                   if pg["name"] not in ["inn_network", "deformation_code"]]
            
            # 添加新的INN网络和变形码参数组
            inn_lr = training_args.inn_lr_init * self.spatial_lr_scale if hasattr(training_args, 'inn_lr_init') else 0.001
            deform_lr = training_args.deformation_code_lr_init * self.spatial_lr_scale if hasattr(training_args, 'deformation_code_lr_init') else 0.001
            
            new_param_groups = non_inn_param_groups + [
                {'params': self.inn_network.parameters(), 'lr': inn_lr, "name": "inn_network"},
                {'params': [self.deformation_code], 'lr': deform_lr, "name": "deformation_code"}
            ]
            
            # 创建新的优化器
            self.optimizer = torch.optim.Adam(new_param_groups, lr=0.0, eps=1e-15)
            
            # 恢复与非INN参数相关的优化器状态
            # 注意：这只是一个简化版，实际实现可能需要更复杂的状态恢复逻辑
            for i, pg in enumerate(non_inn_param_groups):
                if i < len(old_param_groups):
                    for param in pg["params"]:
                        param_id = id(param)
                        if param_id in old_opt_state["state"]:
                            self.optimizer.state[param] = old_opt_state["state"][param_id]
        else:
            # 如果之前没有优化器，重新完整设置
            self.training_setup(training_args)
        
        # 设置学习率调度器
        if hasattr(training_args, 'inn_lr_init') and hasattr(training_args, 'inn_lr_final'):
            self.inn_network_scheduler_args = get_expon_lr_func(
                lr_init=training_args.inn_lr_init * self.spatial_lr_scale,
                lr_final=training_args.inn_lr_final * self.spatial_lr_scale,
                lr_delay_mult=training_args.inn_lr_delay_mult if hasattr(training_args, 'inn_lr_delay_mult') else 0.0,
                max_steps=training_args.inn_lr_max_steps if hasattr(training_args, 'inn_lr_max_steps') else 20000
            )
            
        if hasattr(training_args, 'deformation_code_lr_init') and hasattr(training_args, 'deformation_code_lr_final'):
            self.deformation_code_scheduler_args = get_expon_lr_func(
                lr_init=training_args.deformation_code_lr_init * self.spatial_lr_scale,
                lr_final=training_args.deformation_code_lr_final * self.spatial_lr_scale,
                lr_delay_mult=training_args.deformation_code_lr_delay_mult if hasattr(training_args, 'deformation_code_lr_delay_mult') else 0.0,
                max_steps=training_args.deformation_code_lr_max_steps if hasattr(training_args, 'deformation_code_lr_max_steps') else 20000
            )

    def capture(self, old_version=False):
        """
        捕获当前模型的完整状态，用于后续恢复。
        返回一个包含所有必要参数的元组。
        """
        # 基本属性
        basic_attrs = (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._weights,  # 添加顶点权重
        )
        
        # 透明度和缩放相关属性
        opacity_attrs = (
            self._opacity if not self.use_cell_opacity else torch.empty(0),
            self._cell_o if self.use_cell_opacity else torch.empty(0),
            self._cell_scale,
            self.render_tet_mask,
            self._cell_rotation if not old_version and self.optimizable_rotation else torch.empty(0),
        )
        
        # 变形网络相关属性(可选)
        deformation_attrs = (
            self.freeze,
        )
        
        # 层级结构数据
        hierarchy_attrs = (
            list(self.level_bary_coords),         # 将ParameterList转换为列表存储
            self.level_parent_vertex_indices,   # 层级父顶点索引列表
            self.level_cells,                   # 层级单元拓扑列表
            self.level_cell_lineage_depths,     # 层级单元谱系深度列表
        )
        
        # 层级状态变量
        hierarchy_state = (
            self.num_root_vertices,
            self.global_vertex_count,
            self.current_lod_depth,
            self.absolute_max_lineage_depth,
        )
        
        # 统计和训练相关数据
        training_attrs = (
            self.max_radii2D,
            self.cell_gradient_accum,
            self.cell_gradient_accum_abs,
            self.denom,
            self._init_volume,
            self.optimizer.state_dict() if self.optimizer else None,
            self.spatial_lr_scale,
            self.use_cell_opacity,
            self.use_cell_scale,
            self.optimizable_rotation if not old_version else False,
        )
        
        # 如果有形变网络，也保存它们
        if self.inn_network is not None and self.deformation_code is not None:
            inn_attrs = (
                self.inn_network,
                self.deformation_code,
            )
            return basic_attrs + opacity_attrs + deformation_attrs + hierarchy_attrs + hierarchy_state + training_attrs + inn_attrs
        else:
            return basic_attrs + opacity_attrs + deformation_attrs + hierarchy_attrs + hierarchy_state + training_attrs
    
    def restore(self, model_args, training_args=None, old_version=False):
        """
        从 capture 方法返回的元组中恢复模型状态。
        
        Args:
            model_args: capture 方法返回的元组
            training_args: 可选的训练参数，用于设置优化器
            old_version: 是否为旧版本的保存格式（不包含旋转参数）
            
        Returns:
            无返回值，直接修改当前模型状态
        """
        # 确定是否包含形变网络
        has_inn = len(model_args) > 26  # 基于元组的预期长度判断是否包含形变网络
        
        # 元组索引计数
        idx = 0
        
        # 提取基本属性 - basic_attrs (5个元素)
        self.active_sh_degree = model_args[idx]; idx += 1
        self._xyz = model_args[idx]; idx += 1
        self._features_dc = model_args[idx]; idx += 1
        self._features_rest = model_args[idx]; idx += 1
        self._weights = model_args[idx]; idx += 1  # 恢复顶点权重
        
        # 提取透明度和缩放相关属性 - opacity_attrs (4或5个元素，取决于old_version)
        opacity_item = model_args[idx]; idx += 1
        cell_o_item = model_args[idx]; idx += 1
        self._cell_scale = model_args[idx]; idx += 1
        
        # 如果是旧版本，render_tet_mask 和 cell_rotation 可能不存在
        if old_version:
            self.render_tet_mask = None
            self._cell_rotation = torch.empty(0, device=self._xyz.device)
        else:
            self.render_tet_mask = model_args[idx]; idx += 1
            self._cell_rotation = model_args[idx]; idx += 1
            
        # 提取变形网络状态 - deformation_attrs (1个元素)
        self.freeze = model_args[idx]; idx += 1
        
        # 恢复层级结构数据 - hierarchy_attrs (4个元素)
        bary_coords_list = model_args[idx]; idx += 1
        self.level_parent_vertex_indices = model_args[idx]; idx += 1
        self.level_cells = model_args[idx]; idx += 1
        self.level_cell_lineage_depths = model_args[idx]; idx += 1
        
        # 重新构建ParameterList
        self.level_bary_coords = nn.ParameterList([])
        for bary_coord in bary_coords_list:
            if isinstance(bary_coord, nn.Parameter):
                self.level_bary_coords.append(bary_coord)
            else:
                self.level_bary_coords.append(nn.Parameter(bary_coord))
        
        # 恢复层级状态变量 - hierarchy_state (4个元素)
        self.num_root_vertices = model_args[idx]; idx += 1
        self.global_vertex_count = model_args[idx]; idx += 1
        self.current_lod_depth = model_args[idx]; idx += 1
        self.absolute_max_lineage_depth = model_args[idx]; idx += 1
        
        # 恢复统计和训练相关数据 - training_attrs (9或10个元素，取决于old_version)
        self.max_radii2D = model_args[idx]; idx += 1
        self.cell_gradient_accum = model_args[idx]; idx += 1
        self.cell_gradient_accum_abs = model_args[idx]; idx += 1
        self.denom = model_args[idx]; idx += 1
        self._init_volume = model_args[idx]; idx += 1
        optimizer_state = model_args[idx]; idx += 1
        self.spatial_lr_scale = model_args[idx] if model_args[idx] is not None else 0.0; idx += 1
        self.use_cell_opacity = model_args[idx]; idx += 1
        self.use_cell_scale = model_args[idx]; idx += 1
        
        # 恢复optimizable_rotation标志（如果不是旧版本）
        if not old_version:
            self.optimizable_rotation = model_args[idx]; idx += 1
        else:
            self.optimizable_rotation = False
        
        # 根据use_cell_opacity设置透明度属性
        if self.use_cell_opacity:
            self._cell_o = cell_o_item
            self._opacity = torch.empty(0, device=self._xyz.device)  # 不使用顶点透明度
        else:
            self._opacity = opacity_item
            self._cell_o = torch.empty(0, device=self._xyz.device)  # 不使用单元透明度
        
        # 如果有形变网络，也恢复它们 - inn_attrs (2个元素)
        if has_inn:
            self.inn_network = model_args[idx]; idx += 1
            self.deformation_code = model_args[idx]; idx += 1
        else:
            self.inn_network = None
            self.deformation_code = None
        
        # 如果提供了训练参数，重新设置优化器
        if training_args is not None:
            self.training_setup(training_args)
            
            # 如果保存了优化器状态，恢复它
            if optimizer_state is not None:
                self.optimizer.load_state_dict(optimizer_state)

    def get_barycoords(self, depth=None):
        """
        获取归一化后的重心坐标。
        
        Args:
            depth: 可选，指定要获取的特定深度层级，None表示获取所有层级
            
        Returns:
            list: 包含归一化后的重心坐标的列表，每个元素是一个张量 [T, 4]
                 其中第四个坐标是通过1-前三个坐标之和计算得到的
        """
        max_level = len(self.level_bary_coords)
        
        # 检查depth参数有效性
        if depth is not None:
            assert depth >= 0 and depth < max_level, f"深度 {depth} 无效，有效范围为 0 到 {max_level-1}"
            
            # 只处理指定深度的重心坐标 - 现在每个坐标只有3个分量
            bary_coords_3 = self.bary_coords_activation(self.level_bary_coords[depth])  # [T, 3]
            
            # 直接使用原始的前三个坐标
            normalized_coords_3 = bary_coords_3  # [T, 3]
            
            # 计算第四个坐标
            normalized_coords_4 = 1.0 - torch.sum(normalized_coords_3, dim=1, keepdim=True)  # [T, 1]
            
            # 拼接得到完整的重心坐标 [T, 4]
            normalized_coords = torch.cat([normalized_coords_3, normalized_coords_4], dim=1)  # [T, 4]
            
            return [normalized_coords]
        else:
            # 处理所有深度的重心坐标
            normalized_barycoords = []
            for d in range(max_level):
                # 获取原始重心坐标并应用激活函数 - 现在每个坐标只有3个分量
                bary_coords_3 = self.bary_coords_activation(self.level_bary_coords[d])  # [T, 3]
                
                # 直接使用原始的前三个坐标
                normalized_coords_3 = bary_coords_3  # [T, 3]
                
                # 计算第四个坐标
                normalized_coords_4 = 1.0 - torch.sum(normalized_coords_3, dim=1, keepdim=True)  # [T, 1]
                
                # 拼接得到完整的重心坐标 [T, 4]
                normalized_coords = torch.cat([normalized_coords_3, normalized_coords_4], dim=1)  # [T, 4]
                
                normalized_barycoords.append(normalized_coords)
                
            return normalized_barycoords

    def get_all_vertices(self, depth=None):
        """
        获取全部顶点的位置坐标，包括根顶点和所有虚拟点

        Args:
            depth (int, optional): 如果提供，则只计算到该深度的顶点。默认为None（所有顶点）。

        Returns:
            torch.Tensor: 形状为 [N_all, 3] 的顶点坐标张量
        """
        # 如果depth未指定，使用当前的LOD深度
        if depth is None:
            depth = self.current_lod_depth
        
        global_vertices_list = [self._xyz]  # 从根顶点开始
        bary_coords_list = self.get_barycoords(depth=None)  # 返回完整的[P_d, 4]坐标
        # 对深度0到depth-1的每一层计算虚拟点
        for d in range(min(depth, self.current_lod_depth)):
            if d < len(self.level_parent_vertex_indices):
                # 获取该层父顶点索引和重心坐标
                parent_indices = self.level_parent_vertex_indices[d]  # [P_d, 4]
                
                bary_coords = bary_coords_list[d]  # [P_d, 4] 完整坐标
                # 获取所有已经计算好的顶点 (包括根顶点和之前深度的虚拟点)
                all_vertices_so_far = torch.cat(global_vertices_list, dim=0)  # [N_so_far, 3]
                
                # 查找每个虚拟点的4个父顶点坐标
                parent_vertices = all_vertices_so_far[parent_indices]  # [P_d, 4, 3]
                
                # 使用重心坐标插值计算虚拟点坐标
                virtual_points = torch.sum(parent_vertices * bary_coords.unsqueeze(-1), dim=1)  # [P_d, 3]
                
                # 添加到全局顶点列表
                global_vertices_list.append(virtual_points)
        
        # 拼接所有顶点
        return torch.cat(global_vertices_list, dim=0)
    
    # original  code
    def _prune_optimizer(self, mask, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in name:
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
    
    # original  code
    def cat_tensors_to_optimizer(self, tensors_dict, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in name: 
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


    def cat_tensor_to_optimizer_at_indices(self, tensors_dict, name, start_indices):
        """
        在优化器参数的指定索引位置插入单个张量。
        
        Args:
            tensors_dict: 字典，形式为{参数名: tensor}，每个参数名对应一个张量
            name: 要更新的参数组名称列表
            start_indices: 整数，指定张量应该插入的起始索引位置
            
        Returns:
            dict: 更新后的可优化张量字典
        """
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            if group["name"] in name:
                # 获取当前参数组名称对应的张量
                extension_tensor = tensors_dict[group["name"]]
                
                # 获取当前参数和优化器状态
                current_param = group["params"][0]
                current_tensor = current_param.data
                stored_state = self.optimizer.state.get(current_param, None)
                
                # 构建新的扩展张量
                new_tensor = current_tensor.clone()
                
                # 根据索引位置插入张量
                if start_indices == 0:
                    # 在开头插入
                    new_tensor = torch.cat([extension_tensor, new_tensor], dim=0)
                elif start_indices == new_tensor.shape[0]:
                    # 在末尾追加
                    new_tensor = torch.cat([new_tensor, extension_tensor], dim=0)
                else:
                    if start_indices > new_tensor.shape[0]:
                        raise ValueError(f"插入位置索引 {start_indices} 超出当前张量尺寸 {new_tensor.shape[0]}")
                    # 在中间插入
                    new_tensor = torch.cat([
                        new_tensor[:start_indices],
                        extension_tensor,
                        new_tensor[start_indices:]
                    ], dim=0)
                
                # 更新优化器状态
                if stored_state is not None:
                    # 创建新的优化器状态
                    new_state = {}
                    
                    # 更新动量状态
                    if "exp_avg" in stored_state:
                        # 按照与参数相同的方式构建新的exp_avg
                        new_exp_avg = stored_state["exp_avg"].clone()
                        new_exp_avg_sq = stored_state["exp_avg_sq"].clone()
                        
                        zeros_exp_avg = torch.zeros_like(extension_tensor)
                        
                        # 插入零动量
                        if start_indices == 0:
                            new_exp_avg = torch.cat([zeros_exp_avg, new_exp_avg], dim=0)
                            new_exp_avg_sq = torch.cat([zeros_exp_avg, new_exp_avg_sq], dim=0)
                        elif start_indices == new_exp_avg.shape[0]:
                            new_exp_avg = torch.cat([new_exp_avg, zeros_exp_avg], dim=0)
                            new_exp_avg_sq = torch.cat([new_exp_avg_sq, zeros_exp_avg], dim=0)
                        else:
                            if start_indices > new_exp_avg.shape[0]:
                                raise ValueError(f"插入位置索引 {start_indices} 超出当前张量尺寸 {new_exp_avg.shape[0]}")
                            new_exp_avg = torch.cat([
                                new_exp_avg[:start_indices],
                                zeros_exp_avg,
                                new_exp_avg[start_indices:]
                            ], dim=0)
                            new_exp_avg_sq = torch.cat([
                                new_exp_avg_sq[:start_indices],
                                zeros_exp_avg,
                                new_exp_avg_sq[start_indices:]
                            ], dim=0)
                        
                        new_state["exp_avg"] = new_exp_avg
                        new_state["exp_avg_sq"] = new_exp_avg_sq
                    
                    # 拷贝其他状态值
                    for k, v in stored_state.items():
                        if k not in ["exp_avg", "exp_avg_sq"]:
                            new_state[k] = v
                    
                    # 删除旧状态并设置新状态
                    del self.optimizer.state[current_param]
                    new_param = nn.Parameter(new_tensor.requires_grad_(True))
                    group["params"][0] = new_param
                    self.optimizer.state[new_param] = new_state
                else:
                    # 如果没有状态，只更新参数
                    group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors


    def cat_tensor_to_optimizer_at_indices_list(self, tensors_dict, name, start_indices):
        """
        在优化器参数的指定索引位置插入张量。
        
        Args:
            tensors_dict: 字典，形式为{参数名: [张量1, 张量2, ...]}，每个参数名对应的值是一个张量列表
            name: 要更新的参数组名称列表
            start_indices: 列表，指定每个张量应该插入的起始索引位置，已按顺序排列
            
        Returns:
            dict: 更新后的可优化张量字典
        """
        optimizable_tensors = {}
        
        for group in self.optimizer.param_groups:
            if group["name"] in name:
                # 获取当前参数组名称对应的张量列表
                extension_tensors = tensors_dict[group["name"]]
                
                # 确保张量列表长度与start_indices长度相同
                assert len(extension_tensors) == len(start_indices), f"张量列表长度 {len(extension_tensors)} 与索引列表长度 {len(start_indices)} 不匹配"
                
                # 获取当前参数和优化器状态
                current_param = group["params"][0]
                current_tensor = current_param.data
                stored_state = self.optimizer.state.get(current_param, None)
                
                # 构建新的扩展张量
                new_tensor = current_tensor.clone()
                
                # 创建索引-张量对的列表并按索引倒序排序
                index_tensor_pairs = list(zip(start_indices, extension_tensors))
                # 倒序排列（从大索引到小索引）
                index_tensor_pairs.reverse()
                
                # 按照倒序插入每个张量
                for start_idx, extension_tensor in index_tensor_pairs:
                    # 直接使用原始索引，不需要偏移调整
                    if start_idx == 0:
                        # 在开头插入
                        new_tensor = torch.cat([extension_tensor, new_tensor], dim=0)
                    elif start_idx == new_tensor.shape[0]:
                        # 在末尾追加
                        new_tensor = torch.cat([new_tensor, extension_tensor], dim=0)
                    else:
                        if start_idx > new_tensor.shape[0]:
                            raise ValueError(f"插入位置索引 {start_idx} 超出当前张量尺寸 {new_tensor.shape[0]}")
                        # 在中间插入
                        new_tensor = torch.cat([
                            new_tensor[:start_idx],
                            extension_tensor,
                            new_tensor[start_idx:]
                        ], dim=0)
                
                # 更新优化器状态
                if stored_state is not None:
                    # 创建新的优化器状态
                    new_state = {}
                    
                    # 更新动量状态
                    if "exp_avg" in stored_state:
                        # 按照与参数相同的方式构建新的exp_avg
                        new_exp_avg = stored_state["exp_avg"].clone()
                        new_exp_avg_sq = stored_state["exp_avg_sq"].clone()
                        
                        # 同样倒序处理动量
                        for start_idx, extension_tensor in index_tensor_pairs:
                            zeros_exp_avg = torch.zeros_like(extension_tensor)
                            
                            # 插入零动量
                            if start_idx == 0:
                                new_exp_avg = torch.cat([zeros_exp_avg, new_exp_avg], dim=0)
                                new_exp_avg_sq = torch.cat([zeros_exp_avg, new_exp_avg_sq], dim=0)
                            elif start_idx == new_exp_avg.shape[0]:
                                new_exp_avg = torch.cat([new_exp_avg, zeros_exp_avg], dim=0)
                                new_exp_avg_sq = torch.cat([new_exp_avg_sq, zeros_exp_avg], dim=0)
                            else:
                                if start_idx > new_exp_avg.shape[0]:
                                    raise ValueError(f"插入位置索引 {start_idx} 超出当前张量尺寸 {new_exp_avg.shape[0]}")
                                new_exp_avg = torch.cat([
                                    new_exp_avg[:start_idx],
                                    zeros_exp_avg,
                                    new_exp_avg[start_idx:]
                                ], dim=0)
                                new_exp_avg_sq = torch.cat([
                                    new_exp_avg_sq[:start_idx],
                                    zeros_exp_avg,
                                    new_exp_avg_sq[start_idx:]
                                ], dim=0)
                        
                        new_state["exp_avg"] = new_exp_avg
                        new_state["exp_avg_sq"] = new_exp_avg_sq
                    
                    # 拷贝其他状态值
                    for k, v in stored_state.items():
                        if k not in ["exp_avg", "exp_avg_sq"]:
                            new_state[k] = v
                    
                    # 删除旧状态并设置新状态
                    del self.optimizer.state[current_param]
                    new_param = nn.Parameter(new_tensor.requires_grad_(True))
                    group["params"][0] = new_param
                    self.optimizer.state[new_param] = new_state
                else:
                    # 如果没有状态，只更新参数
                    group["params"][0] = nn.Parameter(new_tensor.requires_grad_(True))
                
                optimizable_tensors[group["name"]] = group["params"][0]
        
        return optimizable_tensors


    def reset_opacity(self):
        """
        重置不透明度为较低值。
        如果use_cell_opacity为True，重置单元不透明度；
        否则重置顶点不透明度。
        """
        if self.use_cell_opacity:
            # 重置单元不透明度
            new_opacity = self.inverse_cell_opacity(torch.min(self.cell_opacity_activation(self._cell_o), 
                                                  torch.ones_like(self._cell_o) * 0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(new_opacity, "cell_opacity")
            self._cell_o = optimizable_tensors["cell_opacity"]
        else:
            # 重置顶点不透明度
            new_opacity = self.inverse_opacity_activation(torch.min(self.opacity_activation(self._opacity), 
                                                       torch.ones_like(self._opacity) * 0.01))
            optimizable_tensors = self.replace_tensor_to_optimizer(new_opacity, "opacity")
            self._opacity = optimizable_tensors["opacity"]
            

    def split_tet(self, cell_indices_to_split_at_deepest_lod):
        """
        将选定的四面体进行1-4细分（在当前最深层级）
        
        Args:
            cell_indices_to_split_at_deepest_lod: 要分裂的单元索引，这些索引对应于当前最深层级的单元
                形状为 [N_split]，类型为 torch.LongTensor
                
        Returns:
            None
        """
        # 如果没有要分裂的单元，直接返回
        if cell_indices_to_split_at_deepest_lod.shape[0] == 0:
            return
        
        # 获取当前最深层级的LOD索引
        deepest_lod = self.current_lod_depth
        
        # 获取最深层级的单元拓扑和谱系深度
        deepest_cells = self.level_cells[deepest_lod]  # [T_deepest, 4]
        deepest_lineage_depths = self.level_cell_lineage_depths[deepest_lod]  # [T_deepest]
        
        # 获取要分裂的单元拓扑和谱系深度
        cells_to_split = deepest_cells[cell_indices_to_split_at_deepest_lod]  # [N_split, 4]
        lineage_depths_to_split = deepest_lineage_depths[cell_indices_to_split_at_deepest_lod]  # [N_split]
        
        # 检查这些单元是否超过最大允许深度
        max_depth_exceeded = lineage_depths_to_split >= self.absolute_max_lineage_depth
        if max_depth_exceeded.any():
            valid_split_mask = ~max_depth_exceeded
            cells_to_split = cells_to_split[valid_split_mask]
            lineage_depths_to_split = lineage_depths_to_split[valid_split_mask]
            cell_indices_to_split_at_deepest_lod =  cell_indices_to_split_at_deepest_lod[valid_split_mask]
            if cells_to_split.shape[0] == 0:
                return  # 所有要分裂的单元都超过了最大深度
        
    
        # 创建要拆分的四面体的mask
        cells_to_split_mask = torch.zeros(deepest_cells.shape[0], dtype=torch.bool, device=deepest_cells.device)
        cells_to_split_mask[cell_indices_to_split_at_deepest_lod] = True
    
        split_render_tet_mask = self.render_tet_mask[cell_indices_to_split_at_deepest_lod]
        expanded_render_tet_mask = split_render_tet_mask.repeat_interleave(4, dim=0)
        remaining_render_tet_mask = self.render_tet_mask[~cells_to_split_mask]
        self.render_tet_mask = torch.cat([remaining_render_tet_mask, expanded_render_tet_mask], dim=0)

        # 处理不透明度和缩放参数
        if self.use_cell_opacity:
            # 获取要拆分的四面体的不透明度
            split_cell_opacity = self.inverse_cell_opacity(self.cell_opacity_activation(self._cell_o[cell_indices_to_split_at_deepest_lod]))   # [N_split, 1]
            # 每个四面体拆分为4个，扩展不透明度值
            expanded_opacity = split_cell_opacity.repeat_interleave(4, dim=0)  # [N_split*4, 1]
            
            if hasattr(self, 'optimizer') and self.optimizer is not None:

                # 使用_prune_optimizer移除被拆分的四面体的不透明度
                optimizable_tensors = self._prune_optimizer(~cells_to_split_mask, ["cell_opacity"])
                self._cell_o = optimizable_tensors["cell_opacity"]


                # 使用cat_tensors_to_optimizer添加新的不透明度值
                tensors_dict = {"cell_opacity": expanded_opacity}
                optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict, ["cell_opacity"])
                self._cell_o = optimizable_tensors["cell_opacity"]
                

            else:
                # 没有优化器时直接操作张量
                # 首先移除被拆分的四面体的不透明度
                remaining_opacity = self._cell_o[~cells_to_split_mask]
                # 然后添加扩展的不透明度值
                self._cell_o = nn.Parameter(torch.cat([remaining_opacity, expanded_opacity], dim=0))

        if self.use_cell_scale:
            # 获取要拆分的四面体的缩放因子
            split_cell_scale = self.inverse_scale_activation(self.scale_activation(self._cell_scale[cell_indices_to_split_at_deepest_lod])*0.5)    # [N_split, 1]
            # 每个四面体拆分为4个，扩展缩放值（可以稍微缩小以保持一致的体积）
            # 拆分为4个后，每个子四面体体积约为原来的1/4，所以缩放因子约为原来的1/4^(1/3)=0.63
            
            expanded_scale = split_cell_scale.repeat_interleave(4, dim=0)   # [N_split*4, 1]
            
            if hasattr(self, 'optimizer') and self.optimizer is not None:

                                
                # 使用_prune_optimizer移除被拆分的四面体的缩放因子
                optimizable_tensors = self._prune_optimizer(~cells_to_split_mask, ["cell_scale"])
                self._cell_scale = optimizable_tensors["cell_scale"]


                # 使用cat_tensors_to_optimizer添加新的缩放值
                tensors_dict = {"cell_scale": expanded_scale}
                optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict, ["cell_scale"])
                self._cell_scale = optimizable_tensors["cell_scale"]

            else:
                # 没有优化器时直接操作张量
                # 首先移除被拆分的四面体的缩放因子
                remaining_scale = self._cell_scale[~cells_to_split_mask]
                # 然后添加扩展的缩放值
                self._cell_scale = nn.Parameter(torch.cat([remaining_scale, expanded_scale], dim=0))
                
        # 处理旋转参数（如果启用）
        if self.optimizable_rotation and self._cell_rotation.shape[0] > 0:
            # 获取要拆分的四面体的旋转
            split_cell_rotation = self._cell_rotation[cell_indices_to_split_at_deepest_lod]  # [N_split, 4]
            # 每个四面体拆分为4个，直接复制旋转值
            expanded_rotation = split_cell_rotation.repeat_interleave(4, dim=0)  # [N_split*4, 4]
            
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # 使用_prune_optimizer移除被拆分的四面体的旋转
                optimizable_tensors = self._prune_optimizer(~cells_to_split_mask, ["cell_rotation"])
                self._cell_rotation = optimizable_tensors["cell_rotation"]
                
                # 使用cat_tensors_to_optimizer添加新的旋转值
                tensors_dict = {"cell_rotation": expanded_rotation}
                optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict, ["cell_rotation"])
                self._cell_rotation = optimizable_tensors["cell_rotation"]
            else:
                # 没有优化器时直接操作张量
                # 首先移除被拆分的四面体的旋转
                remaining_rotation = self._cell_rotation[~cells_to_split_mask]
                # 然后添加扩展的旋转值
                self._cell_rotation = nn.Parameter(torch.cat([remaining_rotation, expanded_rotation], dim=0))
        
        # 计算分裂后会达到的最大深度
        max_new_depth = (lineage_depths_to_split + 1).max().item() if cells_to_split.shape[0] > 0 else 0
        
        # 决定是扩展现有层级还是创建新层级
        # 只有当分裂后的深度超过当前层级结构中包含的深度时，才创建新层级
        if max_new_depth > deepest_lod:
            print(f"创建新层级：从LOD {deepest_lod} 升级到 LOD {deepest_lod+1}")
            # 复制前一层级的单元作为新层级的初始内容
            # 更新current_lod_depth
            self.current_lod_depth += 1
            deepest_lod = self.current_lod_depth
        
        # 为每个要分裂的单元创建子四面体
        N_split = cells_to_split.shape[0]
        
        # 1. 创建新的虚拟顶点作为四面体的中心
        # 只存储前三个重心坐标 [0.25, 0.25, 0.25]，第四个坐标(0.25)通过计算得到
        new_bary_coords_raw = torch.ones((N_split, 3), device=cells_to_split.device) * 0.25
        new_bary_coords = self.inverse_bary_coords_activation(new_bary_coords_raw)
        
        # 2. 为新顶点设置父顶点索引（即原四面体的四个顶点）
        new_parent_vertex_indices = cells_to_split
        
        # 按照约定处理新顶点的索引
        # 首先确定新虚拟顶点的深度 - 这应该是父单元深度+1
        new_virtual_points_depth = lineage_depths_to_split + 1  # [N_split]
        
        # 确保level_bary_coords和level_parent_vertex_indices有足够的深度
        max_vp_depth = new_virtual_points_depth.max().item()
        
        # 记录需要添加到优化器的新参数
        newly_added_params = []
        
        while len(self.level_bary_coords) < max_vp_depth:
            # 创建空参数 - 只需要存储3个坐标
            empty_param = nn.Parameter(torch.empty((0, 3), device=cells_to_split.device))
            self.level_bary_coords.append(empty_param)
            self.level_parent_vertex_indices.append(torch.empty((0, 4), dtype=torch.long, device=cells_to_split.device))
            
            # 记录新参数的索引
            level_idx = len(self.level_bary_coords) - 1
            
            # 如果有优化器，将新参数添加到优化器中
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # 获取学习率 - 使用与其他bary_coords相同的学习率
                lr = 0.001  # 默认学习率
                for group in self.optimizer.param_groups:
                    if group["name"].startswith("level_bary_coords_"):
                        lr = group["lr"]
                        break
                
                # 添加新参数到优化器
                self.optimizer.add_param_group({
                    'params': [empty_param],
                    'lr': lr,
                    "name": f"level_bary_coords_{level_idx}"
                })
                print(f"添加了新的空重心坐标参数到优化器: level_bary_coords_{level_idx}")
                
                # 记录新添加的参数
                newly_added_params.append(level_idx)
        
        # 按深度处理新顶点，从最深层开始倒序处理
        # 首先，创建一个数组来记录每个深度新增的顶点数量
        new_points_count_by_depth = torch.zeros(max_vp_depth + 1, dtype=torch.long, device=cells_to_split.device)
        for d in range(1,max_vp_depth + 1):
            depth_mask = (new_virtual_points_depth == d)
            if depth_mask.any():
                new_points_count_by_depth[d] = depth_mask.sum().item()
        
        # 预先计算每个深度的起始索引（根顶点之后）
        level_start_indices = [0,self.num_root_vertices]  # 第一个索引是根顶点数量
        for d in range(1,max_vp_depth + 1):
            level_start_indices.append(level_start_indices[-1] + self.level_bary_coords[d-1].shape[0])

        # 准备记录新顶点的全局索引
        new_vertex_global_indices = torch.zeros(N_split, dtype=torch.long, device=cells_to_split.device)
        
        # 预处理列表，用于存储新的顶点属性


        # 从深到浅处理每个深度的新顶点
        for d in range(max_vp_depth, 0, -1):
            # 找出当前深度的所有新顶点
            depth_mask = (new_virtual_points_depth == d)
            if not depth_mask.any():
                continue
                
            # 获取当前深度的新顶点数量
            n_new_pts_at_depth = new_points_count_by_depth[d]
            
            # 计算当前深度的起始索引和已有点数
            start_index = level_start_indices[d]  # 深度d的起始索引
            existing_vp_at_depth = self.level_bary_coords[d-1].shape[0]  # 深度d已有的点数
            
            # 计算新顶点的全局索引
            indices_at_depth = torch.arange(
                start_index + existing_vp_at_depth,
                start_index + existing_vp_at_depth + n_new_pts_at_depth,
                dtype=torch.long,
                device=cells_to_split.device
            )
            
            cat_tensor_start_indices = indices_at_depth[0].item()
            # 更新new_vertex_global_indices
            new_vertex_global_indices[depth_mask] = indices_at_depth
            
            tensors_dict = {}
            # 获取父顶点索引
            parent_indices = new_parent_vertex_indices[depth_mask]
            
            # 计算新的属性 - 根据父顶点属性和重心坐标
            # tensors_dict["weights"] = self._weights[parent_indices,:].mean(dim=1)
            tensors_dict["weights"] = self.inverse_weight_activation(self.weight_activation(self._weights[parent_indices,:]).mean(dim=1))
            tensors_dict["f_dc"] = (self._features_dc[parent_indices,:] * 0.25).sum(dim=1)
            tensors_dict["f_rest"] = (self._features_rest[parent_indices,:] * 0.25).sum(dim=1)
            if not self.use_cell_opacity:
                tensors_dict["opacity"] = self.inverse_opacity_activation(self.opacity_activation(self._opacity[parent_indices]).mean(dim=1))
           
           # 使用cat_tensor_to_optimizer_at_indices更新参数和优化器
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # 准备参数名称列表
                param_names = ["weights", "f_dc", "f_rest"]
                if not self.use_cell_opacity:
                    param_names.append("opacity")
                
                # 一次性更新所有参数
                optimizable_tensors = self.cat_tensor_to_optimizer_at_indices(tensors_dict, param_names, cat_tensor_start_indices)
                
                self._weights = optimizable_tensors["weights"]
                self._features_dc = optimizable_tensors["f_dc"]
                self._features_rest = optimizable_tensors["f_rest"]
                if not self.use_cell_opacity:
                    self._opacity = optimizable_tensors["opacity"]
            else:
                # 如果没有优化器，直接拼接
                self._weights = nn.Parameter(torch.cat([
                    self._weights[:cat_tensor_start_indices],
                    tensors_dict["weights"],
                    self._weights[cat_tensor_start_indices:]
                ], dim=0))
                
                self._features_dc = nn.Parameter(torch.cat([
                    self._features_dc[:cat_tensor_start_indices],
                    tensors_dict["f_dc"],
                    self._features_dc[cat_tensor_start_indices:]
                ], dim=0))
                
                self._features_rest = nn.Parameter(torch.cat([
                    self._features_rest[:cat_tensor_start_indices],
                    tensors_dict["f_rest"],
                    self._features_rest[cat_tensor_start_indices:]
                ], dim=0))
                
                if not self.use_cell_opacity and "opacity" in tensors_dict:
                    self._opacity = nn.Parameter(torch.cat([
                        self._opacity[:cat_tensor_start_indices],
                        tensors_dict["opacity"],
                        self._opacity[cat_tensor_start_indices:]
                    ], dim=0))
                
                print(f"没有优化器，直接拼接顶点属性参数")
                        
                    
           
            # 将新虚拟点的重心坐标添加到对应深度
            bary_coords_to_add = new_bary_coords[depth_mask]
            
            # 使用cat_tensors_to_optimizer更新参数和优化器
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                # 准备tensors_dict
                tensors_dict = {f"level_bary_coords_{d-1}": bary_coords_to_add}
                
                # 更新参数和优化器
                optimizable_tensors = self.cat_tensors_to_optimizer(tensors_dict, [f"level_bary_coords_{d-1}"])
                
                # 更新level_bary_coords
                if f"level_bary_coords_{d-1}" in optimizable_tensors:
                    self.level_bary_coords[d-1] = optimizable_tensors[f"level_bary_coords_{d-1}"]
            else:
                # 如果没有优化器，直接拼接
                self.level_bary_coords[d-1] = nn.Parameter(torch.cat([
                    self.level_bary_coords[d-1],
                    bary_coords_to_add
                ], dim=0))
                print(f"没有优化器，直接拼接重心坐标参数: level_bary_coords_{d-1}")
            # 更新父顶点索引
            self.level_parent_vertex_indices[d-1] = torch.cat([
                self.level_parent_vertex_indices[d-1],
                parent_indices
            ], dim=0)
        
            # 获取下一个深度层级的起始索引（添加新点前）
            # 这是我们需要更新的索引的阈值
            next_level_start_index = level_start_indices[d+1]
            
            # 更新所有更深层级的parent_vertex_indices
            for deeper_d in range(d + 1, len(self.level_parent_vertex_indices)+1):
                # 获取更深层级的父顶点索引
                deeper_parents = self.level_parent_vertex_indices[deeper_d-1]
                
                # 对于每个后续深度层，找出其父顶点索引中大于等于阈值的索引
                # 这些索引需要加上当前深度增加的点数量
                indices_to_update = deeper_parents >= next_level_start_index
                
                # 将这些索引值增加当前深度新增的点数
                if indices_to_update.any():
                    self.level_parent_vertex_indices[deeper_d-1][indices_to_update] += n_new_pts_at_depth
                    
        # 更新总顶点计数
        self.global_vertex_count += N_split
        self.update_level_cells()

        n_cells = self.level_cells[-1].shape[0]
        # 更新其他量
        self.denom = torch.zeros((n_cells, 1), device="cuda")
        self.max_radii2D = torch.zeros((n_cells), device="cuda")
        self.cell_gradient_accum = torch.zeros((n_cells, 1), device="cuda")
        self.cell_gradient_accum_abs = torch.zeros((n_cells, 1), device="cuda")
        
    def prune_only(self, min_opacity):
        prune_mask = self.get_opacities(lod=self.current_lod_depth) < min_opacity
        
        
        # 更新render_tet_mask
        self.render_tet_mask[prune_mask] = False

        n_cells = self.level_cells[-1].shape[0]
        # 更新其他量
        self.denom = torch.zeros((n_cells, 1), device="cuda")
        self.max_radii2D = torch.zeros((n_cells), device="cuda")
        self.cell_gradient_accum = torch.zeros((n_cells, 1), device="cuda")
        self.cell_gradient_accum_abs = torch.zeros((n_cells, 1), device="cuda")

    def densify_and_prune(self, viewpoint_camera, max_grad, min_opacity, extent, max_screen_size):
        grads = self.cell_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.cell_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        padded_grad = torch.zeros((self.level_cells[-1].shape[0]), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= max_grad, True, False)
        cell_indices_to_split_at_deepest_lod = torch.nonzero(selected_pts_mask).squeeze(-1)

        prune_mask = self.get_opacities(lod=self.current_lod_depth) < min_opacity
        
        
        # 更新render_tet_mask
        self.render_tet_mask[prune_mask] = False

        self.densify_and_split(cell_indices_to_split_at_deepest_lod)



        # selected_pts_mask[100:,] = False
        # import torchvision
        # rgb = render_htet(viewpoint_camera,self,torch.tensor([1.0,1.0,1.0],device=self._xyz.device),lod=0,cell_mask=selected_pts_mask)['render'].clamp(0.0,1.0)
        # torchvision.utils.save_image(rgb, "/home/juyonggroup/kevin2000/Desktop/gaussian-splatting/scene/debug0.png")
        # mask1 = torch.zeros(self.level_cells[-1].shape[0], device="cuda",dtype=torch.bool)
        # mask1[17027:17099]=True
        # rgb = render_htet(viewpoint_camera,self,torch.tensor([1.0,1.0,1.0],device=self._xyz.device),lod=1,cell_mask=mask1)['render'].clamp(0.0,1.0)
        # torchvision.utils.save_image(rgb, "/home/juyonggroup/kevin2000/Desktop/gaussian-splatting/scene/debug1.png")
        torch.cuda.empty_cache()

    def densify_and_split(self, cell_indices_to_split_at_deepest_lod, max_depth=None):
        """
        根据梯度信息和阈值判断哪些单元需要分裂，并执行分裂操作
        
        Args:
            grads: 当前最精细LOD的每个单元的梯度，形状为 [T_finest]
            grad_threshold: 梯度阈值，高于此值的单元会被分裂
            max_depth: 可选，最大深度限制，默认使用self.absolute_max_lineage_depth
        """
        if max_depth is None:
            max_depth = self.absolute_max_lineage_depth
        
        # 获取当前最深层级索引
        finest_lod = self.current_lod_depth

        
        self.split_tet(cell_indices_to_split_at_deepest_lod)


    def update_parent_indices(self, new_points_count_by_depth):
        """
        更新所有层级的父顶点索引(level_parent_vertex_indices)，考虑到每个深度新增的顶点数量
        
        Args:
            new_points_count_by_depth: 每个深度新增的顶点数量，形状为 [max_depth + 1]
            
        Returns:
            None, 直接修改 self.level_parent_vertex_indices
        """
        max_depth = len(new_points_count_by_depth) - 1
        
        # 预先计算每个深度的起始索引（根顶点之后）
        level_start_indices = [self.num_root_vertices]  # 第一个索引是根顶点数量
        for d in range(max_depth + 1):
            if d < len(self.level_bary_coords):
                level_start_indices.append(level_start_indices[-1] + self.level_bary_coords[d].shape[0])
            else:
                level_start_indices.append(level_start_indices[-1])
        
        # 从深到浅倒序处理，先处理最深的层级
        for d in range(max_depth, -1, -1):
            # 如果当前深度没有新增顶点，跳过
            if new_points_count_by_depth[d] == 0:
                continue
            
            # 获取当前深度的新顶点数量
            n_new_pts_at_depth = new_points_count_by_depth[d]
            
            # 获取下一个深度层级的起始索引（添加新点前）
            # 这是我们需要更新的索引的阈值
            next_level_start_index = level_start_indices[d+1]
            
            # 更新所有更深层级的parent_vertex_indices
            for deeper_d in range(d + 1, len(self.level_parent_vertex_indices)):
                # 获取更深层级的父顶点索引
                deeper_parents = self.level_parent_vertex_indices[deeper_d]
                
                # 对于每个后续深度层，找出其父顶点索引中大于等于阈值的索引
                # 这些索引需要加上当前深度增加的点数量
                indices_to_update = deeper_parents >= next_level_start_index
                
                # 将这些索引值增加当前深度新增的点数
                if indices_to_update.any():
                    self.level_parent_vertex_indices[deeper_d][indices_to_update] += n_new_pts_at_depth
                    
        # 返回更新后的顶点总数
        return self.global_vertex_count


    def update_level_cells(self):
        """
        根据初始四面体拓扑和细分历史重建层级结构中的level_cells
        
        该函数通过从深度0开始，逐层向上构建层级结构：
        1. 深度0的四面体是初始拓扑
        2. 对于每个深度d，判断哪些四面体被细分（通过检查level_parent_vertex_indices）
        3. 对于被细分的四面体，在深度d+1添加四个子四面体
        4. 未被细分的四面体会被保留到下一层级
        
        该实现利用广播机制和索引匹配，显著提高效率
        """
        # 确保至少有一个初始层级
        if len(self.level_cells) == 0 or self.level_cells[0].shape[0] == 0:
            print("没有四面体可重建")
            return
        

        # 保存初始层级的四面体
        initial_cells = self.level_cells[0].clone()
        initial_cells_sort = initial_cells.sort(dim=-1)[0]

        # 重置所有层级结构，只保留初始层级
        initial_cells_count = initial_cells.shape[0]
        self.level_cells = [initial_cells_sort]
        self.level_cell_lineage_depths = [torch.zeros(initial_cells_count, dtype=torch.long, device=self._xyz.device)]
        
        # 计算需要构建的最大深度
        max_depth = len(self.level_parent_vertex_indices)

        print(f"根据细分历史构建层级结构，最大深度: {max_depth}")
        
        # 计算虚拟点的indices
        level_start_indices = [self.num_root_vertices]
        # 逐层构建，从深度0开始
        for d in range(max_depth):
            # 如果已达到当前的最大层级，添加新层级
            lod = d + 1  # 当前lod层级 level_cells只储存lod-1的单元
            if lod >= len(self.level_cells):
                self.level_cells.append(torch.empty((0, 4), dtype=torch.long, device=self._xyz.device))
                self.level_cell_lineage_depths.append(torch.empty(0, dtype=torch.long, device=self._xyz.device))
                level_start_indices.append(level_start_indices[-1]+self.level_parent_vertex_indices[d].shape[0])
            # 获取当前深度的四面体
            current_cells = self.level_cells[d]
            current_depths = self.level_cell_lineage_depths[d]
            
            # 1. 首先，将所有当前层级的四面体复制到下一层级（默认所有四面体都不细分）
            next_cells = current_cells.clone()
            next_depths = current_depths.clone()
            
            # 2. 检查哪些四面体被细分了
            if d < len(self.level_parent_vertex_indices) and self.level_parent_vertex_indices[d].shape[0] > 0:
                # 获取深度d的虚拟点的父顶点索引
                virtual_point_parents = self.level_parent_vertex_indices[d]  # [P_d, 4]
                # sort virtual_points_parents
                virtual_point_parents = virtual_point_parents.sort(dim=-1)[0]
                # 检查哪些四面体被分裂了（通过利用广播机制比较level_parent_vertex_indices和next_cells）

                # split_mask = virtual_point_parents[None,...] == next_cells.view(-1, 1, 4)
                # split_mask = split_mask.all(dim=2).any(dim=-1) 
                # 批处理大小
                batch_size = 100000  # 根据你的GPU内存调整
                split_mask = torch.zeros(next_cells.shape[0], dtype=torch.bool, device=next_cells.device)

                for i in range(0, next_cells.shape[0], batch_size):
                    # 处理一批四面体
                    end_idx = min(i + batch_size, next_cells.shape[0])
                    batch_cells = next_cells[i:end_idx]
                    
                    # 对当前批次进行比较
                    batch_mask = (virtual_point_parents[None,...] == batch_cells.view(-1, 1, 4))
                    batch_mask = batch_mask.all(dim=2).any(dim=-1)
                    
                    # 更新结果
                    split_mask[i:end_idx] = batch_mask
                    
                    # 释放临时张量
                    del batch_mask
                    torch.cuda.empty_cache()
                # 删除被分裂的tetrahedra
                next_cells = next_cells[~split_mask]
                next_depths = next_depths[~split_mask]


                # 计算虚拟点的indices
                virtual_point_indices = torch.arange(level_start_indices[d], level_start_indices[d+1], dtype=torch.long, device=self._xyz.device)

                # 虚拟点indices和四面体任意三个点组成新四面体
                template = torch.tensor([[0,1,2], [0,1,3], [0,2,3], [1,2,3]], dtype=torch.long, device=self._xyz.device)

                # 取virtual_point_parents中任意三个点
                new_cells = torch.cat([virtual_point_parents[:,template],virtual_point_indices[:, None, None].expand(-1, 4, -1)],dim=-1)
                new_cells = new_cells.view(-1,4)
                # 将新四面体添加到next_cells中
                next_cells = torch.cat([next_cells, new_cells], dim=0)
                next_depths = torch.cat([next_depths, torch.ones(new_cells.shape[0], dtype=torch.long, device=self._xyz.device) * lod], dim=0)

                self.level_cells[lod] = next_cells
                self.level_cell_lineage_depths[lod] = next_depths

        self.level_cells[0] = initial_cells
        # 打印每个层级的单元数量
        # for d in range(len(self.level_cells)):
        #     print(f"层级 {d} 有 {self.level_cells[d].shape[0]} 个单元")

    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.cell_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.cell_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        

    def get_tet_vertices(self, lod, tet_indices):
        """
        获取指定四面体的顶点坐标
        
        Args:
            lod: 指定的LOD层级
            tet_indices: 四面体在level_cells[lod]中的索引，形状为[N]
            
        Returns:
            torch.Tensor: 形状为[N, 4, 3]的张量，包含N个四面体的4个顶点坐标
        """
        # 获取四面体顶点索引
        tets = self.level_cells[lod][tet_indices]  # [N, 4]
        
        # 获取所有顶点坐标
        all_vertices = self.get_all_vertices()  # [V, 3]
        
        # 提取四面体顶点坐标
        tet_vertices = all_vertices[tets]  # [N, 4, 3]
        
        return tet_vertices

    def save_interactive_plot(self, lod=0, output_path="tet_mesh2.html"):
        """
        使用 Plotly 创建可交互的 3D 可视化并保存为 HTML 文件
        
        Args:
            lod (int): 要可视化的LOD级别，默认为0（最粗糙的网格）
            output_path (str): 保存HTML文件的路径
            
        Returns:
            str: 保存的文件路径
        """
        import numpy as np
        import plotly.graph_objects as go
        import os
        
        # 确保请求的LOD在有效范围内
        if lod < 0 or lod > self.current_lod_depth:
            raise ValueError(f"LOD必须在 0 到 {self.current_lod_depth} 之间，当前请求的LOD为 {lod}")
        
        # 获取当前LOD的顶点和单元
        vertices = self.get_vertices(lod).detach().cpu().numpy()
        cells = self.level_cells[lod].detach().cpu().numpy()
        
        # 为每个四面体生成随机颜色
        np.random.seed(42)  # 设置随机种子以保持颜色一致性
        cell_colors = np.random.rand(len(cells), 3)  # 生成随机RGB颜色
        
        # 创建四面体的面（每个四面体有4个三角形面）
        faces = []
        list_of_face_colors = []  # 存储每个面的颜色
        for i, cell in enumerate(cells):
            # 四面体的四个面
            faces.extend([
                [cell[0], cell[1], cell[2]],  # 面1
                [cell[0], cell[1], cell[3]],  # 面2
                [cell[0], cell[2], cell[3]],  # 面3
                [cell[1], cell[2], cell[3]]   # 面4
            ])
            # 将RGB颜色转换为字符串格式
            color_str = f'rgb({int(cell_colors[i][0]*255)}, {int(cell_colors[i][1]*255)}, {int(cell_colors[i][2]*255)})'
            list_of_face_colors.extend([color_str] * 4)
            
        # 创建四面体的边（每个四面体有6条边）
        edges = []
        for cell in cells:
            # 四面体的六条边
            edges.extend([
                [cell[0], cell[1]],  # 边1
                [cell[0], cell[2]],  # 边2
                [cell[0], cell[3]],  # 边3
                [cell[1], cell[2]],  # 边4
                [cell[1], cell[3]],  # 边5
                [cell[2], cell[3]]   # 边6
            ])
        
        # 创建3D网格
        fig = go.Figure()
        
        # 添加面
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=[face[0] for face in faces],
            j=[face[1] for face in faces],
            k=[face[2] for face in faces],
            opacity=0.7,
            facecolor=list_of_face_colors,  # 使用 facecolor 属性为每个面指定颜色
            flatshading=True,
            showscale=False
        ))
        
        # 添加边线
        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=[vertices[edge[0]][0], vertices[edge[1]][0]],
                y=[vertices[edge[0]][1], vertices[edge[1]][1]],
                z=[vertices[edge[0]][2], vertices[edge[1]][2]],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))
        
        # 设置布局
        fig.update_layout(
            title=f'Tetrahedral Mesh (LOD {lod})',
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            showlegend=False
        )
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # 保存为HTML文件
        fig.write_html(output_path)
        print(f"已保存交互式可视化到: {output_path}")
        
        return output_path
    
    def compute_quality_metrics(self):
        vertices = self.get_vertices(0)
        tets = self.level_cells[0]
        rms_edge_ratio = compute_rms_edge_ratio(vertices, tets)
        print(f"RMS edge ratio: {rms_edge_ratio}")
        return rms_edge_ratio
    
    def compute_quality_loss(self, threshold=0.8):
        vertices = self.get_vertices(0)
        tets = self.level_cells[0]
        rms_edge_ratio = compute_rms_edge_ratio(vertices, tets)

        return  (torch.max(threshold - rms_edge_ratio, torch.zeros_like(rms_edge_ratio)) ** 2).mean()



    def signed_volume_tet(self):
        vertices = self.get_vertices(0)
        tets = self.level_cells[0]

        return  compute_tetrahedron_signed_volume(vertices, tets)
    
    def signed_volume_tet_loss(self):
        return torch.relu(-self.signed_volume_tet()).mean()
    

    def get_lod_cell_data(self):
        """
        根据当前LOD层级，获取该层级的有关cell的数据，例如cell_opacity, cell_scale
        """
        # 确保层级结构已构建

       

        current_lod_to_finest_start_indices =  [-1 * torch.ones((self.level_cells[d].shape[0]),device=self._xyz.device,dtype=torch.long) for d in range(len(self.level_cells))]
        current_lod_to_finest_end_indices =  [-1 * torch.ones((self.level_cells[d].shape[0]),device=self._xyz.device,dtype=torch.long) for d in range(len(self.level_cells))]
        
        current_lod_to_finest_start_indices[self.current_lod_depth] = torch.arange(self.num_root_vertices,device=self._xyz.device,dtype=torch.long)
        current_lod_to_finest_end_indices[self.current_lod_depth] = torch.arange(self.num_root_vertices,device=self._xyz.device,dtype=torch.long)
        # 从最细粒度层级开始，逐层向上构建映射关系
        cell_opacity_list = [None for i in range(len(self.level_cells)-1)] + [self._cell_o]
        cell_scale_list = [None for i in range(len(self.level_cells)-1)] + [self._cell_scale]

        for d in range(len(self.level_cells),1,-1):
            lod = d -1
            cell_to_split = self.level_parent_vertex_indices[lod - 1]
            cell_to_split_sort = cell_to_split.sort(dim=-1)[0]
            batch_size = 100000  # 根据你的GPU内存调整
            next_cell_opacity = cell_opacity_list[lod]
            next_cell_scale = cell_scale_list[lod]
            pre_cell_opacity = torch.zeros((self.level_cells[lod-1].shape[0], next_cell_opacity.shape[1]), device=next_cell_opacity.device, dtype=next_cell_opacity.dtype)
            pre_cell_scale = torch.zeros((self.level_cells[lod-1].shape[0], next_cell_scale.shape[1]), device=next_cell_scale.device, dtype=next_cell_scale.dtype)
            split_mask = torch.zeros(self.level_cells[lod-1].shape[0], dtype=torch.bool, device=next_cell_scale.device)


            for i in range(0, pre_cell_scale.shape[0], batch_size):
                # 处理一批四面体
                end_idx = min(i + batch_size, pre_cell_scale.shape[0])
                batch_cells = self.level_cells[lod-1][i:end_idx].sort(dim=-1)[0]
                
                # 对当前批次进行比较
                batch_mask = (cell_to_split_sort[None,...] == batch_cells.view(-1, 1, 4))
                batch_mask = batch_mask.all(dim=2).any(dim=-1)
                
                # 更新结果
                split_mask[i:end_idx] = batch_mask
                
                # 释放临时张量
                del batch_mask
                torch.cuda.empty_cache()
            num_split = split_mask.sum()
            assert num_split == cell_to_split.shape[0]
            num_not_split = (~split_mask).sum()
            pre_cell_opacity[~split_mask] = next_cell_opacity[:num_not_split]
            pre_cell_scale[~split_mask] = next_cell_scale[:num_not_split]
            # argmax_opacity = next_cell_opacity[num_not_split:].view(-1,4,1).m(dim=1)[0]
            pre_cell_opacity[split_mask] = next_cell_opacity[num_not_split:].view(-1,4,1).max(dim=1)[0]
            pre_cell_scale[split_mask] = self.inverse_scale_activation(1.58*self.scale_activation(next_cell_scale[num_not_split:].view(-1,4,1).max(dim=1)[0]))

            cell_opacity_list[lod-1] = pre_cell_opacity
            cell_scale_list[lod-1] = pre_cell_scale

        self._cell_o_list = cell_opacity_list
        self._cell_scale_list = cell_scale_list
        

            
        # 获取最细粒度层级的索引
        finest_level = len(self.level_cells) - 1
      
        return 0

if __name__ == '__main__':
   pass