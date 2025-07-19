import torch

def compute_signed_volume(v0, v1, v2, v3):
    """
    计算四面体的有向体积。

    参数:
    - v0, v1, v2, v3: 张量形状为 [N, 3]，表示四面体的四个顶点。

    返回:
    - signed_volumes: 张量形状为 [N]，表示每个四面体的有向体积。
    """
    return torch.einsum('bi,bi->b', (v1 - v0).cross(v2 - v0), (v3 - v0)) / 6.0

vertices = torch.tensor([
    [0.0, 0.0, 0.0],  # v0
    [1.0, 0.0, 0.0],  # v1
    [0.0, 1.0, 0.0],  # v2
    [0.0, 0.0, 1.0]   # v3
], dtype=torch.float32)

# 定义四面体单元，顶点索引按照正向排列
cells = torch.tensor([
    [0,1,3,2]
], dtype=torch.long)  # [1,4]

# 计算原始四面体的有向体积
original_volume = compute_signed_volume(
    vertices[cells[0, 0]].unsqueeze(0),
    vertices[cells[0, 1]].unsqueeze(0),
    vertices[cells[0, 2]].unsqueeze(0),
    vertices[cells[0, 3]].unsqueeze(0)
)

print(f"原始四面体的有向体积: {original_volume.item()}")


def subdivide_tetrahedron(cells, vertices):
    """
    细分四面体，将每个四面体分割成八个子四面体。

    参数:
    - cells: 张量形状为 [M, 4]，包含要细分的四面体顶点索引。
    - vertices: 张量形状为 [V, 3]，包含所有顶点的坐标。

    返回:
    - new_cells: 张量形状为 [M*8, 4]，包含细分后的所有四面体顶点索引。
    - new_vertices: 张量形状为 [V + K, 3]，包含新增的顶点坐标。
    """
    edge_offsets = torch.tensor([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3],
        [2, 3]
    ], dtype=torch.long)  # [6,2]

    edges = cells[:, edge_offsets]  # [M,6,2]
    # 对每条边的顶点索引进行排序，以便识别唯一边
    edges_sorted, _ = torch.sort(edges, dim=-1)  # [M,6,2]
    edges_reshaped = edges_sorted.reshape(-1, 2)  # [M*6,2]
    unique_edges, inverse_indices = torch.unique(edges_reshaped, dim=0, sorted=True, return_inverse=True)  # [K,2]

    # 计算每条唯一边的中点坐标
    v1 = vertices[unique_edges[:, 0]]  # [K,3]
    v2 = vertices[unique_edges[:, 1]]  # [K,3]
    new_xyz = (v1 + v2) / 2.0  # [K,3]

    # 添加新的顶点到顶点列表
    new_vertices = torch.cat([vertices, new_xyz], dim=0)  # [V + K, 3]
    num_existing_vertices = vertices.shape[0]
    new_vertex_indices = torch.arange(num_existing_vertices, num_existing_vertices + unique_edges.shape[0], dtype=torch.long)  # [K]

    # 获取新的顶点索引
    mid_indices = new_vertex_indices[inverse_indices].reshape(-1, 6)  # [M,6]

    # 提取原始顶点索引
    v0 = cells[:, 0].unsqueeze(1)  # [M,1]
    v1_orig = cells[:, 1].unsqueeze(1)  # [M,1]
    v2_orig = cells[:, 2].unsqueeze(1)  # [M,1]
    v3_orig = cells[:, 3].unsqueeze(1)  # [M,1]

    # 获取中点顶点索引
    m01 = mid_indices[:, 0].unsqueeze(1)  # [M,1]
    m02 = mid_indices[:, 1].unsqueeze(1)  # [M,1]
    m03 = mid_indices[:, 2].unsqueeze(1)  # [M,1]
    m12 = mid_indices[:, 3].unsqueeze(1)  # [M,1]
    m13 = mid_indices[:, 4].unsqueeze(1)  # [M,1]
    m23 = mid_indices[:, 5].unsqueeze(1)  # [M,1]

    # 创建新的八个子四面体
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

    return new_cells, new_vertices

new_cells, new_vertices = subdivide_tetrahedron(cells, vertices)

print(f"细分后生成的四面体数量: {new_cells.shape[0]}")

# 计算细分后四面体的有向体积
v0_new = new_vertices[new_cells[:, 0]]
v1_new = new_vertices[new_cells[:, 1]]
v2_new = new_vertices[new_cells[:, 2]]
v3_new = new_vertices[new_cells[:, 3]]

signed_volumes_new = compute_signed_volume(v0_new, v1_new, v2_new, v3_new)


print(signed_volumes_new)