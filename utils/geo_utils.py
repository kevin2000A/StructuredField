import torch
import torch.nn.functional as F
import time
import numpy as np
import math


EPSILON = 1e-10

def scaled_jacobian_def1(verts):
    """
    计算 Scaled Jacobian (定义 1: 基于单一顶点的三条边长)。
    理想值为 1/√2 (~0.707)，乘以 sqrt(2) 可归一化到 1。

    Args:
        verts (torch.Tensor): 四面体顶点坐标，形状为 (N, 4, 3)。

    Returns:
        torch.Tensor: 每个四面体的 Scaled Jacobian 值，形状为 (N,)。
        torch.Tensor: 对应的 det(J) 值，形状为 (N,)。
    """
    if verts.ndim != 3 or verts.shape[1] != 4 or verts.shape[2] != 3:
        raise ValueError("Input verts tensor must have shape (N, 4, 3)")

    p1 = verts[:, 0, :]  # (N, 3)
    p2 = verts[:, 1, :]  # (N, 3)
    p3 = verts[:, 2, :]  # (N, 3)
    p4 = verts[:, 3, :]  # (N, 3)

    # 从 p1 出发的边向量
    e1 = p2 - p1  # (N, 3)
    e2 = p3 - p1  # (N, 3)
    e3 = p4 - p1  # (N, 3)

    # 构造雅可比矩阵 (行向量为边)
    # J 的形状为 (N, 3, 3)
    J = torch.stack([e1, e2, e3], dim=1)

    # 计算雅可比行列式 det(J)
    det_J = torch.det(J) # (N,)

    # 计算三条边向量的长度
    len_e1 = torch.linalg.norm(e1, dim=1) # (N,)
    len_e2 = torch.linalg.norm(e2, dim=1) # (N,)
    len_e3 = torch.linalg.norm(e3, dim=1) # (N,)

    # 计算分母
    denominator = len_e1 * len_e2 * len_e3

    # 计算 Scaled Jacobian (定义 1)
    # 添加 EPSILON 防止除以零
    sj = det_J / (denominator + EPSILON)

    # 如果需要归一化到 1:
    # sj_normalized = sj * np.sqrt(2.0)

    return sj, det_J

def scaled_jacobian_def2(verts):
    """
    计算 Scaled Jacobian (定义 2: 基于体积和 RMS 边长)。
    理想值为 1。

    Args:
        verts (torch.Tensor): 四面体顶点坐标，形状为 (N, 4, 3)。

    Returns:
        torch.Tensor: 每个四面体的 Scaled Jacobian 值，形状为 (N,)。
        torch.Tensor: 对应的 det(J) 值，形状为 (N,)。
    """
    if verts.ndim != 3 or verts.shape[1] != 4 or verts.shape[2] != 3:
        raise ValueError("Input verts tensor must have shape (N, 4, 3)")

    p1 = verts[:, 0, :]  # (N, 3)
    p2 = verts[:, 1, :]  # (N, 3)
    p3 = verts[:, 2, :]  # (N, 3)
    p4 = verts[:, 3, :]  # (N, 3)

    # --- 计算 det(J) ---
    e1 = p2 - p1
    e2 = p3 - p1
    e3 = p4 - p1
    J = torch.stack([e1, e2, e3], dim=1)
    det_J = torch.det(J) # (N,)


    # --- 计算 RMS 边长 ---
    # 另外三条边向量
    e4 = p3 - p2 # (N, 3)
    e5 = p4 - p2 # (N, 3)
    e6 = p4 - p3 # (N, 3)

    # 计算所有 6 条边长的平方
    lsq1 = torch.sum(e1**2, dim=1) # (N,)
    lsq2 = torch.sum(e2**2, dim=1) # (N,)
    lsq3 = torch.sum(e3**2, dim=1) # (N,)
    lsq4 = torch.sum(e4**2, dim=1) # (N,)
    lsq5 = torch.sum(e5**2, dim=1) # (N,)
    lsq6 = torch.sum(e6**2, dim=1) # (N,)

    # 计算 RMS 边长的平方
    lrms_sq = (lsq1 + lsq2 + lsq3 + lsq4 + lsq5 + lsq6) / 6.0 # (N,)

    # 计算 RMS 边长
    # 添加 EPSILON 防止 sqrt(0) 和后续除零
    lrms = torch.sqrt(lrms_sq + EPSILON) # (N,)

    # 计算 Scaled Jacobian (定义 2)
    # SJ = sqrt(2) * det(J) / L_rms^3
    # 添加 EPSILON 防止除以零
    sj = np.sqrt(2.0) * det_J / (lrms**3 + EPSILON)

    return sj, det_J


def compute_triangle_areas(vertices, triangles):
    """计算三角形面积

    Args:
        vertices (torch.Tensor): 形状为 [V, 3] 的顶点坐标
        triangles (torch.Tensor): 形状为 [F, 3] 的三角形面片索引

    Returns:
        torch.Tensor: 形状为 [F] 的三角形面积
    """
    # 获取三角形三个顶点
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    
    # 计算两条边向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 计算叉积
    cross = torch.cross(edge1, edge2, dim=1)
    
    # 计算面积（叉积长度的一半）
    areas = 0.5 * torch.norm(cross, dim=1)
    
    return areas

def tetrahedra_to_triangles(tetrahedra):
    """
    将四面体索引转换为三角形面片索引

    Args:
        tetrahedra (torch.Tensor): 形状为 [T, 4] 的四面体索引

    Returns:
        torch.Tensor: 形状为 [4*T, 3] 的三角形面片索引
    """
    device = tetrahedra.device
    T = tetrahedra.shape[0]
    
    # 定义四面体的4个面的顶点索引
    faces_template = torch.tensor([
        [0, 1, 2],  # 面0
        [0, 2, 3],  # 面1
        [0, 3, 1],  # 面2
        [1, 3, 2]   # 面3
    ], device=device, dtype=torch.long)
    
    # 复制每个四面体的索引为4个面
    tet_indices = torch.arange(T, device=device).unsqueeze(1).repeat(1, 4).view(-1)
    
    # 创建三角形面片索引
    triangles = torch.zeros((T*4, 3), dtype=torch.long, device=device)
    
    for i in range(4):
        face_indices = faces_template[i]
        triangles[i::4] = tetrahedra[:, face_indices]
    
    return triangles


def compute_tetrahedron_signed_volume(vertices, tetrahedra):
    """Compute volume of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Volumes with shape [T]
    """
    # Extract vertices of each tetrahedron
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # Compute edge vectors
    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0
    
    # Compute volume using triple product
    volume = torch.sum(torch.cross(v10, v20, dim=1) * v30, dim=1) / 6.0
    
    return volume



def compute_tetrahedron_volume(vertices, tetrahedra):
    """Compute volume of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Volumes with shape [T]
    """
    # Extract vertices of each tetrahedron
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # Compute edge vectors
    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0
    
    # Compute volume using triple product
    volume = torch.sum(torch.cross(v10, v20, dim=1) * v30, dim=1) / 6.0
    
    return volume

def compute_tetrahedron_surface_area(vertices, tetrahedra):
    """Compute surface area of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Surface areas with shape [T]
    """
    # Extract vertices of each tetrahedron
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # Compute areas of the four triangular faces
    face_areas = torch.zeros((tetrahedra.shape[0], 4), device=vertices.device)
    
    # Face 0-1-2
    a = torch.norm(v1 - v0, dim=1)
    b = torch.norm(v2 - v1, dim=1)
    c = torch.norm(v0 - v2, dim=1)
    s = (a + b + c) / 2.0
    face_areas[:, 0] = torch.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Face 0-1-3
    a = torch.norm(v1 - v0, dim=1)
    b = torch.norm(v3 - v1, dim=1)
    c = torch.norm(v0 - v3, dim=1)
    s = (a + b + c) / 2.0
    face_areas[:, 1] = torch.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Face 0-2-3
    a = torch.norm(v2 - v0, dim=1)
    b = torch.norm(v3 - v2, dim=1)
    c = torch.norm(v0 - v3, dim=1)
    s = (a + b + c) / 2.0
    face_areas[:, 2] = torch.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Face 1-2-3
    a = torch.norm(v2 - v1, dim=1)
    b = torch.norm(v3 - v2, dim=1)
    c = torch.norm(v1 - v3, dim=1)
    s = (a + b + c) / 2.0
    face_areas[:, 3] = torch.sqrt(s * (s - a) * (s - b) * (s - c))
    
    # Sum up face areas
    surface_area = torch.sum(face_areas, dim=1)
    
    return surface_area

def compute_tetrahedron_edge_lengths(vertices, tetrahedra):
    """计算四面体的6条边长

    Args:
        vertices (torch.Tensor): 形状为 [V, 3] 的顶点坐标
        tetrahedra (torch.Tensor): 形状为 [T, 4] 的四面体索引

    Returns:
        torch.Tensor: 形状为 [T, 6] 的四面体边长
    """
    # 获取四面体的四个顶点
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # 计算6条边的长度
    edge01 = torch.norm(v1 - v0, dim=1)
    edge02 = torch.norm(v2 - v0, dim=1)
    edge03 = torch.norm(v3 - v0, dim=1)
    edge12 = torch.norm(v2 - v1, dim=1)
    edge13 = torch.norm(v3 - v1, dim=1)
    edge23 = torch.norm(v3 - v2, dim=1)
    
    # 将6条边长度堆叠在一起
    edge_lengths = torch.stack([edge01, edge02, edge03, edge12, edge13, edge23], dim=1)
    
    return edge_lengths

def compute_tetrahedron_circumradius(vertices, tetrahedra):
    """Compute circumradius (radius of circumscribed sphere) of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Circumradii with shape [T]
    """
    # Extract vertices of each tetrahedron
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # Compute edge vectors
    v10 = v1 - v0
    v20 = v2 - v0
    v30 = v3 - v0
    
    # Compute volume
    volume = torch.abs(torch.sum(torch.cross(v10, v20, dim=1) * v30, dim=1)) / 6.0
    
    # Compute edge lengths
    a = torch.norm(v1 - v0, dim=1)
    b = torch.norm(v2 - v0, dim=1)
    c = torch.norm(v3 - v0, dim=1)
    d = torch.norm(v2 - v1, dim=1)
    e = torch.norm(v3 - v1, dim=1)
    f = torch.norm(v3 - v2, dim=1)
    
    # Compute squared edge lengths
    a2, b2, c2 = a**2, b**2, c**2
    d2, e2, f2 = d**2, e**2, f**2
    
    # Compute determinant using formula from computational geometry
    det = torch.zeros_like(volume)
    det = det + a2 * (e2 * f2 - d2 * c2)
    det = det - b2 * (a2 * f2 - c2 * e2)
    det = det + c2 * (a2 * d2 - b2 * e2)
    det = det - d2 * (b2 * c2 - a2 * f2)
    det = det + e2 * (a2 * c2 - b2 * f2)
    det = det - f2 * (a2 * b2 - c2 * d2)
    
    # # Compute circumradius
    # # For numerical stability, we use a different formula for regular tetrahedra
    # edge_sum = a + b + c + d + e + f
    # edge_mean = edge_sum / 6.0
    # is_regular = ((a - edge_mean).abs() < 1e-6) & \
    #              ((b - edge_mean).abs() < 1e-6) & \
    #              ((c - edge_mean).abs() < 1e-6) & \
    #              ((d - edge_mean).abs() < 1e-6) & \
    #              ((e - edge_mean).abs() < 1e-6) & \
    #              ((f - edge_mean).abs() < 1e-6)
    
    # # For regular tetrahedron, the formula is R = (sqrt(6) / 4) * edge_length
    # circumradius_regular = (math.sqrt(6) / 4.0) * edge_mean
    
    # For arbitrary tetrahedron, use the determinant formula
    circumradius_arbitrary = torch.sqrt(torch.abs(det)) / (24.0 * volume)
    
    # # Choose the appropriate formula based on tetrahedron regularity
    # circumradius = torch.where(is_regular, circumradius_regular, circumradius_arbitrary)
    
    return circumradius_arbitrary

def compute_tetrahedron_inradius(volume, surface_area):
    """Compute inradius (radius of inscribed sphere) of tetrahedra
    
    Args:
        volume (torch.Tensor): Volumes with shape [T]
        surface_area (torch.Tensor): Surface areas with shape [T]
        
    Returns:
        torch.Tensor: Inradii with shape [T]
    """
    # r = 3 * V / S
    inradius = 3.0 * volume / surface_area
    
    return inradius

def compute_tetrahedron_face_normals(vertices, tetrahedra, normalize=True):
    """计算四面体面的法向量

    Args:
        vertices (torch.Tensor): 形状为 [V, 3] 的顶点坐标
        tetrahedra (torch.Tensor): 形状为 [T, 4] 的四面体索引
        normalize (bool, optional): 是否归一化法向量. 默认 True.

    Returns:
        torch.Tensor: 形状为 [T, 4, 3] 的四面体面法向量
    """
    # 获取四面体的四个顶点
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # 计算四个面的法向量
    # 面0: [v0, v1, v2]
    n0 = torch.cross(v1 - v0, v2 - v0, dim=1)
    
    # 面1: [v0, v2, v3]
    n1 = torch.cross(v2 - v0, v3 - v0, dim=1)
    
    # 面2: [v0, v3, v1]
    n2 = torch.cross(v3 - v0, v1 - v0, dim=1)
    
    # 面3: [v1, v3, v2]
    n3 = torch.cross(v3 - v1, v2 - v1, dim=1)
    
    # 堆叠法向量
    normals = torch.stack([n0, n1, n2, n3], dim=1)
    
    # 归一化法向量
    if normalize:
        norms = torch.norm(normals, dim=2, keepdim=True)
        normals = normals / (norms + 1e-10)
    
    return normals

def compute_tetrahedron_dihedral_angles(vertices, tetrahedra):
    """计算四面体的6个二面角

    Args:
        vertices (torch.Tensor): 形状为 [V, 3] 的顶点坐标
        tetrahedra (torch.Tensor): 形状为 [T, 4] 的四面体索引

    Returns:
        torch.Tensor: 形状为 [T, 6] 的四面体二面角（弧度）
    """
    # 计算面的法向量
    face_normals = compute_tetrahedron_face_normals(vertices, tetrahedra, normalize=True)
    
    # 计算二面角（每对相邻面之间的角度）
    # 在四面体中有6对相邻面（对应6条边）
    
    # 边对应的面对
    face_pairs = [
        (0, 1),  # 面0和面1（对应边v0-v2）
        (0, 2),  # 面0和面2（对应边v0-v1）
        (0, 3),  # 面0和面3（对应边v1-v2）
        (1, 2),  # 面1和面2（对应边v0-v3）
        (1, 3),  # 面1和面3（对应边v2-v3）
        (2, 3)   # 面2和面3（对应边v1-v3）
    ]
    
    dihedral_angles = []
    
    for i, j in face_pairs:
        # 获取相邻面的法向量
        ni = face_normals[:, i]
        nj = face_normals[:, j]
        
        # 计算法向量夹角的余弦值
        cos_angle = torch.sum(ni * nj, dim=1)
        
        # 裁剪余弦值到有效范围 [-1, 1]，避免数值误差
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        # 计算二面角（为了便于梯度计算，我们使用arccos）
        angle = torch.acos(cos_angle)
        
        # 计算二面角的补角 (180° - angle)，这是更常用的定义
        angle = torch.tensor(math.pi, device=vertices.device) - angle
        
        dihedral_angles.append(angle)
    
    return torch.stack(dihedral_angles, dim=1)

def compute_tetrahedron_jacobian(vertices, tetrahedra, reference_tet=None):
    """计算从参考四面体到目标四面体的雅可比矩阵

    Args:
        vertices (torch.Tensor): 形状为 [V, 3] 的顶点坐标
        tetrahedra (torch.Tensor): 形状为 [T, 4] 的四面体索引
        reference_tet (torch.Tensor, optional): 参考四面体的4个顶点坐标 [4, 3]，默认为单位四面体

    Returns:
        torch.Tensor: 形状为 [T, 3, 3] 的雅可比矩阵
    """
    device = vertices.device
    
    # 获取当前四面体的四个顶点
    v0 = vertices[tetrahedra[:, 0]]
    v1 = vertices[tetrahedra[:, 1]]
    v2 = vertices[tetrahedra[:, 2]]
    v3 = vertices[tetrahedra[:, 3]]
    
    # 如果指定了自定义参考四面体，则需要计算完整的变换
    if reference_tet is not None:
        # 获取参考四面体的边矩阵
        ref_edges = torch.stack([
            reference_tet[1] - reference_tet[0],
            reference_tet[2] - reference_tet[0],
            reference_tet[3] - reference_tet[0]
        ], dim=1)  # [3, 3]
        
        # 计算参考边矩阵的逆
        ref_edges_inv = torch.inverse(ref_edges)
        
        # 计算当前四面体的边矩阵
        edges = torch.stack([
            v1 - v0,
            v2 - v0,
            v3 - v0
        ], dim=2)  # [T, 3, 3]
        
        # 计算雅可比矩阵 J = edges * ref_edges_inv
        J = torch.matmul(edges, ref_edges_inv)
    else:
        # 默认参考四面体时，ref_edges_inv是单位矩阵，可直接使用edges作为雅可比矩阵
        J = torch.stack([
            v1 - v0,
            v2 - v0,
            v3 - v0
        ], dim=2)  # [T, 3, 3]
    
    return J

# ================== 质量度量函数 ==================

def compute_radius_ratio(vertices, tetrahedra):
    """Compute radius ratio (quality metric) of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Radius ratios with shape [T], in range (0, 1]
        where 1 is a regular tetrahedron and values approaching 0 are degenerate
    """
    # Compute volume and surface area
    volume = compute_tetrahedron_volume(vertices, tetrahedra)
    surface_area = compute_tetrahedron_surface_area(vertices, tetrahedra)
    
    # Compute inradius and circumradius
    inradius = compute_tetrahedron_inradius(volume, surface_area)
    circumradius = compute_tetrahedron_circumradius(vertices, tetrahedra)
    
    # Compute radius ratio (normalized to [0, 1])
    # Formula: ρ = r / R * 3
    # For regular tetrahedron, r/R = 1/3, so ρ = 1
    radius_ratio = inradius / (circumradius + 1e-10) * 3.0
    
    return radius_ratio

def compute_rms_edge_ratio(vertices, tetrahedra):
    """Compute RMS edge length ratio (quality metric) of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: RMS edge ratios with shape [T], in range (0, 1]
        where 1 is a regular tetrahedron and values approaching 0 are degenerate
    """
    # Compute volume
    volume = compute_tetrahedron_volume(vertices, tetrahedra)
    
    # Compute edge lengths
    edges = compute_tetrahedron_edge_lengths(vertices, tetrahedra)
    
    # Compute RMS edge length
    l_rms = torch.sqrt(torch.mean(edges**2, dim=1))
    
    # Compute RMS edge ratio (normalized to [0, 1])
    # Formula: Q = c * V / l_rms^3
    # where c = 8.48528... for regular tetrahedron to make Q = 1
    normalization_factor = 8.48528137423857
    rms_ratio = normalization_factor * volume / (l_rms**3 + 1e-10)
    # print("actual rms_ratio: ", rms_ratio)
    # Clamp to [0, 1] to handle numerical issues
    
    return rms_ratio

def compute_min_dihedral_angle(vertices, tetrahedra):
    """Compute minimum dihedral angle of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Minimum dihedral angles in degrees with shape [T]
    """
    angles = compute_tetrahedron_dihedral_angles(vertices, tetrahedra)
    min_angle = torch.min(angles, dim=1)[0]
    return min_angle

def compute_min_sine(vertices, tetrahedra):
    """Compute minimum sine of dihedral angles of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Minimum sine values with shape [T]
    """
    angles_deg = compute_tetrahedron_dihedral_angles(vertices, tetrahedra)
    angles_rad = angles_deg * (math.pi / 180.0)
    sines = torch.sin(angles_rad)
    min_sine = torch.min(sines, dim=1)[0]
    return min_sine

def compute_jacobian_condition_number(vertices, tetrahedra):
    """Compute Jacobian condition number of tetrahedra
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        
    Returns:
        torch.Tensor: Condition numbers with shape [T]
    """
    # Compute Jacobian matrices
    jacobian = compute_tetrahedron_jacobian(vertices, tetrahedra)
    
    # For regular tetrahedron reference, the Jacobian is the identity matrix
    # scaled by a constant (for unit tetrahedron).
    # The condition number is ||J||_F * ||J^{-1}||_F
    

    jacobian_inv = torch.inverse(jacobian)

    
    # Compute Frobenius norms
    j_norm = torch.norm(jacobian, p='fro', dim=(1, 2))
    j_inv_norm = torch.norm(jacobian_inv, p='fro', dim=(1, 2))
    
    # Compute condition number
    condition_number = j_norm * j_inv_norm
    
    # For a regular tetrahedron, the condition number is 3.0
    # For any other tetrahedron, it is > 3.0
    # For numerical stability, ensure no number is less than 3.0
    condition_number = torch.clamp(condition_number, min=3.0)
    
    return condition_number

# ================== 损失函数 ==================

def radius_ratio_loss(vertices, tetrahedra, target=0.5, weight=1.0):
    """Compute radius ratio loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        target (float, optional): Target radius ratio. Default 0.5.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    radius_ratio = compute_radius_ratio(vertices, tetrahedra)
    return threshold_loss(radius_ratio, target, weight)

def exp_radius_ratio_loss(vertices, tetrahedra, alpha=5.0, weight=1.0):
    """Compute exponential decay radius ratio loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        alpha (float, optional): Decay rate. Default 5.0.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    radius_ratio = compute_radius_ratio(vertices, tetrahedra)
    return exp_decay_loss(radius_ratio, alpha, weight)

def inverse_radius_ratio_loss(vertices, tetrahedra, epsilon=1e-6, weight=1.0):
    """Compute inverse radius ratio loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        epsilon (float, optional): Small value to prevent division by zero. Default 1e-6.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    radius_ratio = compute_radius_ratio(vertices, tetrahedra)
    return inverse_loss(radius_ratio, epsilon, weight)

def rms_edge_ratio_loss(vertices, tetrahedra, target=0.5, weight=1.0):
    """Compute RMS edge ratio loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        target (float, optional): Target RMS edge ratio. Default 0.5.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    rms_ratio = compute_rms_edge_ratio(vertices, tetrahedra)
    return threshold_loss(rms_ratio, target, weight)

def dihedral_angle_loss(vertices, tetrahedra, min_angle=45.0, weight=1.0):
    """Compute dihedral angle loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        min_angle (float, optional): Minimum desired dihedral angle in degrees. Default 45.0.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    min_dihedral = compute_min_dihedral_angle(vertices, tetrahedra)
    # Normalize to [0, 1] range for consistent weighting
    # Maximum possible angle in a tetrahedron is ~70.53 degrees
    normalized_angle = min_dihedral / 70.53
    return threshold_loss(normalized_angle, min_angle / 70.53, weight)

def min_sine_loss(vertices, tetrahedra, target=0.5, weight=1.0):
    """Compute minimum sine loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        target (float, optional): Target minimum sine value. Default 0.5.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    min_sine = compute_min_sine(vertices, tetrahedra)
    return threshold_loss(min_sine, target, weight)

def jacobian_condition_number_loss(vertices, tetrahedra, target=5.0, weight=1.0):
    """Compute Jacobian condition number loss
    
    Args:
        vertices (torch.Tensor): Vertices with shape [V, 3]
        tetrahedra (torch.Tensor): Tetrahedron indices with shape [T, 4]
        target (float, optional): Target condition number. Default 5.0.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    condition_number = compute_jacobian_condition_number(vertices, tetrahedra)
    # For condition number, smaller is better
    # The minimum possible value is 3 for a regular tetrahedron
    # Invert the comparison since we want to minimize the condition number
    loss = torch.clamp(condition_number - target, min=0.0)
    return weight * torch.mean(loss)

# Threshold loss function
def threshold_loss(quality_metric, target=0.5, weight=1.0):
    """Compute threshold loss for a quality metric
    
    Args:
        quality_metric (torch.Tensor): Quality metric values
        target (float, optional): Target value. Default 0.5.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    # Loss = max(0, target - quality)
    loss = torch.clamp(target - quality_metric, min=0.0)
    return weight * torch.mean(loss)

# Exponential decay loss function
def exp_decay_loss(quality_metric, alpha=100.0, weight=1.0):
    """Compute exponential decay loss for a quality metric
    
    Args:
        quality_metric (torch.Tensor): Quality metric values
        alpha (float, optional): Decay rate. Default 5.0.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    # Loss = exp(-alpha * quality)
    loss = torch.exp(-alpha * quality_metric)
    return weight * torch.mean(loss)

# Inverse loss function
def inverse_loss(quality_metric, epsilon=1e-6, weight=1.0):
    """Compute inverse loss for a quality metric
    
    Args:
        quality_metric (torch.Tensor): Quality metric values
        epsilon (float, optional): Small value to prevent division by zero. Default 1e-6.
        weight (float, optional): Loss weight. Default 1.0.
        
    Returns:
        torch.Tensor: Loss value
    """
    # Loss = 1 / (quality + epsilon)
    loss = 1.0 / (quality_metric + epsilon)
    return weight * torch.mean(loss) 