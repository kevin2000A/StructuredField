import numpy as np

def create_tetrahedral_grid(scale, Nx, Ny, Nz):
    """
    Generates a tetrahedral grid within a cube defined by [-scale, scale]^3.
    Each voxel in a uniform grid is decomposed into 6 tetrahedra.

    Args:
        scale (float): Defines the cubic region from -scale to scale along X, Y, Z axes.
        Nx (int): Number of voxels (cells) along the X-axis.
        Ny (int): Number of voxels (cells) along the Y-axis.
        Nz (int): Number of voxels (cells) along the Z-axis.

    Returns:
        tuple: (vertices, tetrahedra)
            vertices (np.ndarray): Array of shape (total_vertices, 3) containing world coordinates of unique vertices.
            tetrahedra (np.ndarray): Array of shape (Nx*Ny*Nz*6, 4) containing global vertex IDs for each tetrahedron.
    """
    # 体素尺寸计算
    voxel_size = np.array([2*scale/Nx, 2*scale/Ny, 2*scale/Nz])
    
    # 生成顶点坐标（完全向量化）
    x = np.linspace(-scale, scale, Nx+1)
    y = np.linspace(-scale, scale, Ny+1)
    z = np.linspace(-scale, scale, Nz+1)
    
    # 使用meshgrid生成三维网格坐标（优化内存布局）
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
    vertices = grid.reshape(-1, 3).astype(np.float32)

    # 生成体素索引（完全向量化）
    vx, vy, vz = np.mgrid[:Nx, :Ny, :Nz]
    voxels = np.stack((vx, vy, vz), axis=-1).reshape(-1, 3)

    # 预计算所有体素的8个角点偏移量
    offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.int64)

    # 计算所有体素角点的三维索引（向量化）
    corners_3d = voxels[:, None, :] + offsets[None, :, :]  # [N_voxel, 8, 3]

    # 转换为全局顶点ID（向量化）
    strides = np.array([1, Nx+1, (Nx+1)*(Ny+1)], dtype=np.int64)
    global_ids = (corners_3d * strides).sum(axis=-1)  # [N_voxel, 8]

    # 定义四面体模板（扩展到所有体素）
    tet_template = np.array([
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6]
    ], dtype=np.int64)

    # 生成所有四面体（向量化）
    tets = global_ids[:, tet_template].reshape(-1, 4)

    return vertices, tets

if __name__ == '__main__':
    # Example usage:
    scale = 1.0
    Nx, Ny, Nz = 2, 2, 2 # Create a 2x2x1 grid of voxels

    vertices, tetrahedra = create_tetrahedral_grid(scale, Nx, Ny, Nz)

    print("Generated Vertices:")
    print(vertices)
    print(f"Shape: {vertices.shape}")

    print("\nGenerated Tetrahedra (using global vertex IDs):")
    print(tetrahedra)
    print(f"Shape: {tetrahedra.shape}")

    # Expected number of vertices: (Nx+1)*(Ny+1)*(Nz+1)
    expected_verts = (Nx + 1) * (Ny + 1) * (Nz + 1)
    print(f"\nExpected number of vertices: {expected_verts}, Got: {vertices.shape[0]}")

    # Expected number of tetrahedra: Nx*Ny*Nz*6
    expected_tets = Nx * Ny * Nz * 6
    print(f"Expected number of tetrahedra: {expected_tets}, Got: {tetrahedra.shape[0]}")

    # Save to .npz file
    output_filename = f"tetra_grid_{Nx}x{Ny}x{Nz}_scale{scale}.npz"
    np.savez(output_filename, vertices=vertices, cells=tetrahedra)
    print(f"\nSaved grid to {output_filename}")

    # Further verification (optional):
    # Check if all vertex indices in tetrahedra are valid
    assert np.all(tetrahedra >= 0) and np.all(tetrahedra < vertices.shape[0]), "Invalid vertex indices in tetrahedra!"

    # Check for positive volume for a few tetrahedra (requires a volume calculation function)
    # (This is more involved and depends on vertex ordering for signed volume)
    print("\nBasic checks passed.")

    # 新增meshio保存功能
    try:
        import meshio
    except ImportError:
        print("\nmeshio未安装，无法保存MSH文件。请使用以下命令安装：")
        print("pip install meshio")
    else:
        def save_with_meshio(vertices, cells, filename):
            """使用meshio保存为Gmsh格式"""
            # 创建meshio网格对象
            # 注意：meshio使用0-based索引，但Gmsh需要1-based，需要转换
            mesh = meshio.Mesh(
                points=vertices,
                cells=[("tetra", cells)]  # 转换为1-based索引
            )
            
            # 添加物理组信息（可选）
            mesh.field_data = {"material": np.array([1], dtype=np.int32)}
            
            # 写入文件
            mesh.write(
                filename,
                file_format="gmsh",
                binary=False,
            )

        # 保存为MSH文件
        msh_filename = f"tetra_grid_{Nx}x{Ny}x{Nz}_scale{scale}.msh"
        save_with_meshio(vertices, tetrahedra, msh_filename)
        print(f"\nSaved MSH file to {msh_filename} using meshio")
