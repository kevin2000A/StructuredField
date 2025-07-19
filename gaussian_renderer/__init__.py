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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.tetrahedra_model import TetrahedraModel
from scene.hierarchical_tetrahedra_model import HierarchicalTetrahedraModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    # NaN/inf checks for render function
    if torch.isnan(means3D).any() or torch.isinf(means3D).any():
        raise ValueError("[render] NaN or Inf found in means3D before rasterization!")
    if cov3D_precomp is not None and (torch.isnan(cov3D_precomp).any() or torch.isinf(cov3D_precomp).any()):
        raise ValueError("[render] NaN or Inf found in cov3D_precomp before rasterization!")
    if colors_precomp is not None and (torch.isnan(colors_precomp).any() or torch.isinf(colors_precomp).any()):
        raise ValueError("[render] NaN or Inf found in colors_precomp before rasterization!")
    if opacity is not None and (torch.isnan(opacity).any() or torch.isinf(opacity).any()):
        raise ValueError("[render] NaN or Inf found in opacity before rasterization!")

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}



def render_tet(viewpoint_camera, tets : TetrahedraModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((tets.get_cells.shape[0],4),dtype=tets._xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
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
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    (means3D,   # (T,3)
     cov3D_precomp,   # (T,3,3)
     opacity,  # (T,1)
     weights
    ) = tets.convert_gaussian()
    # means3D = pc.get_xyz
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    # cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    
    shs = None
    colors_precomp = None
 
    shs_view = tets.get_features.transpose(1, 2).view(-1, 3, (tets.max_sh_degree+1)**2)
    dir_pp = (tets.get_xyz() - viewpoint_camera.camera_center.repeat(tets.get_features.shape[0], 1))

    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(tets.active_sh_degree, shs_view, dir_pp_normalized) + 0.5
    
    tet_colors = torch.clamp_min(sh2rgb[tets.get_cells],0.0)
    # m_softplus = torch.nn.Softplus(beta=10)
    # soft_colors = m_softplus(sh2rgb)
    # tet_colors = soft_colors[tets.get_cells]
    colors_precomp = torch.sum(tet_colors * weights, dim=-2) / torch.sum(weights, dim=-2)
    # colors_precomp = torch.clamp_min(tet_colors + 0.5, 0.0)

    
    # NaN/inf checks for render_tet function
    if torch.isnan(means3D).any() or torch.isinf(means3D).any():
        raise ValueError("[render_tet] NaN or Inf found in means3D before rasterization!")
    if cov3D_precomp is not None and (torch.isnan(cov3D_precomp).any() or torch.isinf(cov3D_precomp).any()):
        raise ValueError("[render_tet] NaN or Inf found in cov3D_precomp before rasterization!")
    if colors_precomp is not None and (torch.isnan(colors_precomp).any() or torch.isinf(colors_precomp).any()):
        raise ValueError("[render_tet] NaN or Inf found in colors_precomp before rasterization!")
    if opacity is not None and (torch.isnan(opacity).any() or torch.isinf(opacity).any()):
        raise ValueError("[render_tet] NaN or Inf found in opacity before rasterization!")
    # scales and rotations are None in render_tet, shs is also None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, inv_depth, mask = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = None,
        rotations = None,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "render_alpha": mask,
            "radii": radii}




def render_htet(viewpoint_camera, tets : HierarchicalTetrahedraModel, bg_color : torch.Tensor, lod=0, scaling_modifier = 1.0, override_color = None, cell_mask=None, feature_interp_mode="sh_first"):
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
    
    # NaN/inf checks for render_htet function
    if torch.isnan(means3D).any() or torch.isinf(means3D).any():
        raise ValueError("[render_htet] NaN or Inf found in means3D before rasterization!")
    if cov3D_precomp is not None and (torch.isnan(cov3D_precomp).any() or torch.isinf(cov3D_precomp).any()):
        raise ValueError("[render_htet] NaN or Inf found in cov3D_precomp before rasterization!")
    if colors_precomp is not None and (torch.isnan(colors_precomp).any() or torch.isinf(colors_precomp).any()):
        raise ValueError("[render_htet] NaN or Inf found in colors_precomp before rasterization!")
    if opacities is not None and (torch.isnan(opacities).any() or torch.isinf(opacities).any()):
        raise ValueError("[render_htet] NaN or Inf found in opacities before rasterization!")
    # scales, rotations, and shs are None in render_htet before rasterizer call

    # 光栅化可见的高斯体到图像，获取它们的屏幕半径
    if tets.render_tet_mask is not None:
        cell_mask = tets.render_tet_mask[...,0]
        means3D = means3D[cell_mask]
        means2D = screenspace_points[cell_mask]
        cov3D_precomp = cov3D_precomp[cell_mask]
        opacities = opacities[cell_mask]
        colors_precomp = colors_precomp[cell_mask]
        
    rendered_image, radii, inv_depth, mask = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D_precomp)

    vis_filter = torch.zeros((tets.level_cells[-1].shape[0],),dtype=torch.bool,device=radii.device)
    if tets.render_tet_mask is not None:
        vis_filter[tets.render_tet_mask[...,0]] = True
    # 返回渲染结果
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": vis_filter,
        "render_alpha": mask,
        "radii": radii
    }




def render_htet_multi_lod(viewpoint_camera, tets : HierarchicalTetrahedraModel, bg_color : torch.Tensor, lod=0, scaling_modifier = 1.0, override_color = None, cell_mask=None, feature_interp_mode="sh_first"):
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
    opacities = tets.cell_opacity_activation(tets._cell_o_list[lod])
    
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
    
    # NaN/inf checks for render_htet function
    if torch.isnan(means3D).any() or torch.isinf(means3D).any():
        raise ValueError("[render_htet] NaN or Inf found in means3D before rasterization!")
    if cov3D_precomp is not None and (torch.isnan(cov3D_precomp).any() or torch.isinf(cov3D_precomp).any()):
        raise ValueError("[render_htet] NaN or Inf found in cov3D_precomp before rasterization!")
    if colors_precomp is not None and (torch.isnan(colors_precomp).any() or torch.isinf(colors_precomp).any()):
        raise ValueError("[render_htet] NaN or Inf found in colors_precomp before rasterization!")
    if opacities is not None and (torch.isnan(opacities).any() or torch.isinf(opacities).any()):
        raise ValueError("[render_htet] NaN or Inf found in opacities before rasterization!")
    # scales, rotations, and shs are None in render_htet before rasterizer call

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




# def render_htet_with_reindex(viewpoint_camera, tets : HierarchicalTetrahedraModel, bg_color : torch.Tensor, new_vertiecs, reindices, lod=0, scaling_modifier = 1.0,  feature_interp_mode="sh_first"):
#     """
#     渲染层级四面体模型场景。
    
#     Args:
#         viewpoint_camera: 视点相机
#         tets: HierarchicalTetrahedraModel实例
#         pipe: 渲染管线
#         bg_color: 背景颜色张量（必须在GPU上）
#         lod: 使用的LOD层级，默认为0（最粗糙层级）
#         scaling_modifier: 缩放修改器
#         override_color: 可选的覆盖颜色
#         feature_interp_mode: 特征插值模式，可选值：
#             - "sh_first": 先插值SH系数再转RGB（默认）
#             - "rgb_first": 先转RGB再插值
    
#     Returns:
#         渲染结果字典
#     """
 
#     # 获取当前LOD层级的单元数
#     cells = tets.get_cells(lod)
    
#     # 创建零张量，用于梯度回传（screen-space means）
#     screenspace_points = torch.zeros((cells.shape[0], 4), dtype=tets._xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # 设置光栅化配置
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=tets.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=False
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    
#     # 获取单元中心，协方差和不透明度
#     means3D, cov3D_precomp = tets.get_gs_mean_covs(lod)
#     opacities = tets.cell_opacity_activation(tets._cell_o_list[lod])
    
#     # 获取顶点权重（用于插值）
#     weights = tets.weight_activation(tets._weights)[cells]  # [T, 4, 1]
    
#     # 特征处理
#     if feature_interp_mode == "sh_first":
#         # 路径1: 先插值SH系数再转RGB
#         features = tets.get_gs_features(lod)  # 获取已经插值后的单元SH系数
        
#         # 将特征转换为可用于SH评估的格式
#         features_view = features.view(-1, 3, (tets.max_sh_degree+1)**2)
        
#         # 计算视角方向
#         dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
#         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        
#         # 评估SH并转换为RGB
#         sh2rgb = eval_sh(tets.active_sh_degree, features_view, dir_pp_normalized)
#         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
#     else:  # "rgb_first"
#         # 路径2: 先转RGB再插值
#         # 获取顶点特征和坐标
#         vertex_features = tets.get_vertices_features()  # 获取顶点SH系数
#         vertices = tets.get_vertices(lod)
        
#         # 计算视角方向（对每个顶点）
#         dir_pp = (vertices - viewpoint_camera.camera_center.repeat(vertices.shape[0], 1))
#         dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        
#         # 将顶点特征转换为可用于SH评估的格式
#         features_view = vertex_features.view(-1, 3, (tets.max_sh_degree+1)**2)
        
#         # 评估SH并转换为RGB
#         vertex_colors = eval_sh(tets.active_sh_degree, features_view, dir_pp_normalized)
#         vertex_colors = torch.clamp_min(vertex_colors + 0.5, 0.0)
        
#         # 获取单元顶点的颜色
#         cell_vertex_colors = vertex_colors[cells]  # [T, 4, 3]
        
#         # 使用权重插值顶点颜色得到单元颜色
#         weights_expanded = weights / torch.sum(weights, dim=1, keepdim=True)  # [T, 4, 1]
#         colors_precomp = torch.sum(cell_vertex_colors * weights_expanded, dim=1)  # [T, 3]
    
#     # NaN/inf checks for render_htet function
#     if torch.isnan(means3D).any() or torch.isinf(means3D).any():
#         raise ValueError("[render_htet] NaN or Inf found in means3D before rasterization!")
#     if cov3D_precomp is not None and (torch.isnan(cov3D_precomp).any() or torch.isinf(cov3D_precomp).any()):
#         raise ValueError("[render_htet] NaN or Inf found in cov3D_precomp before rasterization!")
#     if colors_precomp is not None and (torch.isnan(colors_precomp).any() or torch.isinf(colors_precomp).any()):
#         raise ValueError("[render_htet] NaN or Inf found in colors_precomp before rasterization!")
#     if opacities is not None and (torch.isnan(opacities).any() or torch.isinf(opacities).any()):
#         raise ValueError("[render_htet] NaN or Inf found in opacities before rasterization!")
#     # scales, rotations, and shs are None in render_htet before rasterizer call

#     # 光栅化可见的高斯体到图像，获取它们的屏幕半径
#     if cell_mask is not None:
#         means3D = means3D[cell_mask]
#         cov3D_precomp = cov3D_precomp[cell_mask]
#         opacities = opacities[cell_mask]
#         colors_precomp = colors_precomp[cell_mask]
        
#     rendered_image, radii, inv_depth, mask = rasterizer(
#         means3D=means3D,
#         means2D=screenspace_points,
#         shs=None,
#         colors_precomp=colors_precomp,
#         opacities=opacities,
#         scales=None,
#         rotations=None,
#         cov3D_precomp=cov3D_precomp)

    
#     # 返回渲染结果
#     return {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter": radii > 0,
#         "render_alpha": mask,
#         "radii": radii
#     }
