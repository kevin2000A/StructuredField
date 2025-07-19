实现文档：均匀体素网格的六面体单元四面体化
文档版本： 1.0
日期： 2025年5月17日
作者： (你的名字/团队)

1. 目标

本文档描述了将一个定义在 [-scale, scale]^3 立方体区域内的均匀体素（voxel）网格转换为一个连续的四面体网格的实现方法。每个体素将被确定地分解为6个四面体，确保相邻体素间的共享顶点和面得到正确处理，从而形成一个无缝的四面体网格。

2. 输入参数

scale: 定义立方体区域范围的浮点数。区域为 [-scale, scale] 沿X, Y, Z轴。
Nx: 沿X轴的体素（单元）数量 (整数)。
Ny: 沿Y轴的体素（单元）数量 (整数)。
Nz: 沿Z轴的体素（单元）数量 (整数)。
3. 核心算法与数据结构

3.1. 网格与索引定义

体素大小 (Voxel Size):

voxel_size_x = (2 * scale) / Nx
voxel_size_y = (2 * scale) / Ny
voxel_size_z = (2 * scale) / Nz
顶点网格维度 (Vertex Grid Dimensions):

num_vertices_x = Nx + 1
num_vertices_y = Ny + 1
num_vertices_z = Nz + 1
总顶点数: total_vertices = num_vertices_x * num_vertices_y * num_vertices_z
顶点世界坐标 (Vertex World Coordinates):
对于顶点网格中的一个顶点，其三维整数索引为 (ix, iy, iz)，其中 $0 \\le ix \< num\_vertices\_x$, $0 \\le iy \< num\_vertices\_y$, $0 \\le iz \< num\_vertices\_z$:

world_x = -scale + ix * voxel_size_x
world_y = -scale + iy * voxel_size_y
world_z = -scale + iz * voxel_size_z
全局一维顶点索引 (Global 1D Vertex ID):
对于三维顶点索引 (ix, iy, iz):

global_vertex_id = ix + (iy * num_vertices_x) + (iz * num_vertices_x * num_vertices_y)
3.2. 数据结构

全局顶点列表 (Global Vertex List):

类型：浮点数数组或向量列表。
内容：存储所有唯一顶点的世界坐标 (world_x, world_y, world_z)。
大小：total_vertices。
索引：通过上述 global_vertex_id 进行索引。
例如 (Python列表的列表): global_vertices = [[x0,y0,z0], [x1,y1,z1], ...]
全局四面体列表 (Global Tetrahedra List):

类型：整数数组或向量列表。
内容：存储所有四面体。每个四面体由其4个顶点的 global_vertex_id 定义。
大小：Nx * Ny * Nz * 6 个四面体。
例如 (Python列表的列表): global_tetrahedra = [[v_idx1, v_idx2, v_idx3, v_idx4], ...] (注意：顶点的顺序可能影响四面体的朝向/法线，具体参见7.1节)
3.3. 实现步骤

(可选但推荐) 初始化全局顶点列表:
遍历所有可能的顶点三维索引 (ix, iy, iz)：

$0 \\le ix \< num\_vertices\_x$
$0 \\le iy \< num\_vertices\_y$
$0 \\le iz \< num\_vertices\_z$ 计算每个顶点的世界坐标并按其 global_vertex_id 存入 global_vertices 列表。
遍历体素并生成四面体:
遍历所有体素，体素由其最小角点（对应局部顶点 v_0）的三维体素索引 (vx, vy, vz) 标识：

$0 \\le vx \< Nx$
$0 \\le vy \< Ny$
$0 \\le vz \< Nz$
对于每个体素 (vx, vy, vz):

a. 确定体素8个角点的三维顶点索引:
这些角点对应于局部顶点 v_0 到 v_7（见4.1节的定义）。它们的在全局顶点网格中的三维索引 (ix, iy, iz) 如下：

corner_vertex_3d_indices[0] = (vx, vy, vz) // 对应局部 v_0
corner_vertex_3d_indices[1] = (vx + 1, vy, vz) // 对应局部 v_1
corner_vertex_3d_indices[2] = (vx + 1, vy + 1, vz) // 对应局部 v_2
corner_vertex_3d_indices[3] = (vx, vy + 1, vz) // 对应局部 v_3
corner_vertex_3d_indices[4] = (vx, vy, vz + 1) // 对应局部 v_4
corner_vertex_3d_indices[5] = (vx + 1, vy, vz + 1) // 对应局部 v_5
corner_vertex_3d_indices[6] = (vx + 1, vy + 1, vz + 1) // 对应局部 v_6
corner_vertex_3d_indices[7] = (vx, vy + 1, vz + 1) // 对应局部 v_7
b. 计算8个角点的全局一维顶点ID:
对于上述每个 corner_vertex_3d_indices[k] = (ix, iy, iz)，计算其 global_vertex_id:
voxel_corner_global_ids[k] = ix + (iy * num_vertices_x) + (iz * num_vertices_x * num_vertices_y)

c. 应用四面体分解模板生成6个四面体:
使用 voxel_corner_global_ids 和4.2节中定义的分解模板，生成6个四面体。每个四面体由4个 global_vertex_id 组成。
将这6个新生成的四面体添加到 global_tetrahedra 列表中。

4. 四面体分解模板

4.1. 局部顶点定义

考虑一个单位立方体，其8个局部顶点（v_0 到 v_7）定义如下，以确保一致性：

v_0=(0,0,0) (体素的最小x,y,z角点)
v_1=(1,0,0)
v_2=(1,1,0)
v_3=(0,1,0)
v_4=(0,0,1)
v_5=(1,0,1)
v_6=(1,1,1) (体素的最大x,y,z角点, v_0的体对角点)
v_7=(0,1,1)
此局部顶点顺序对应于3.3.2.a步骤中 corner_vertex_3d_indices 的顺序。

4.2. 六四面体分解（沿主对角线 v_0v_6）

以下6个四面体由其局部顶点索引定义。在实现中，这些局部索引将被替换为相应的 voxel_corner_global_ids。

tet1 = (v_0, v_1, v_2, v_6)
tet2 = (v_0, v_2, v_3, v_6)
tet3 = (v_0, v_3, v_7, v_6)
tet4 = (v_0, v_7, v_4, v_6)
tet5 = (v_0, v_4, v_5, v_6)
tet6 = (v_0, v_5, v_1, v_6)