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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud, BasicTetrahedra
import meshio

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    tetrahedra: BasicTetrahedra
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchmsh(path):

    mesh = meshio.read(path)
    tetrahedra = {}
    tetrahedra["vertices"] = np.array(mesh.points, dtype=np.float32)
    
    for cell in mesh.cells:
        if cell.type == "tetra":
            tetrahedra["cells"] = np.array(cell.data, dtype=np.int32)
            break
    return BasicTetrahedra(vertices=tetrahedra["vertices"], cells=tetrahedra["cells"], colors=None)
    
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    tetra_path = os.path.join(path, "points3d.msh")
    print("Using points3d.msh")
    
    try:
        tetra = fetchmsh(tetra_path)
    except:
        tetra = None
        raise ValueError("No tetrahedra found")
        
    scene_info = SceneInfo(point_cloud=pcd,
                           tetrahedra=tetra,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # c2w[:3, 3] *= 1.5
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = Image.fromarray(im_data, "RGBA")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, init_mesh, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if init_mesh:
        tetra_path = os.path.join(path, "points3d_init.msh")
        print("Using points3d_init.msh")
    else:
        tetra_path = os.path.join(path, "points3d.msh")
        print("Using points3d.msh")
    
    if not os.path.exists(tetra_path):
        print("Generating tetrahedra...")
        import torch
        from utils.create_tetra_init import create_tetrahedral_grid
        
        def project_vertices_to_camera(vertices_gpu, camera):
            """
            将顶点投影到相机视角，并判断哪些顶点在前景区域
            
            Args:
                vertices_gpu: torch.Tensor, 形状为[N, 3]的顶点坐标
                camera: CameraInfo对象，包含相机参数和图像
                
            Returns:
                torch.Tensor: 形状为[N]的布尔掩码，表示每个顶点是否在当前相机前景中可见
            """
            # 构建相机投影矩阵 - 从世界坐标到相机坐标
            w2c = np.eye(4)
            w2c[:3, :3] = camera.R  # 转置，因为R是从世界到相机
            w2c[:3, 3] = camera.T
            

            # 计算内参矩阵
            width, height = camera.width, camera.height
            focal_x = 0.5 * width / np.tan(0.5 * camera.FovX)
            focal_y = 0.5 * height / np.tan(0.5 * camera.FovY)
            K = np.array([[focal_x, 0, width/2], 
                          [0, focal_y, height/2], 
                          [0, 0, 1]])
            
            # 转换为PyTorch
            w2c_gpu = torch.tensor(w2c, dtype=torch.float32, device="cuda")
            K_gpu = torch.tensor(K, dtype=torch.float32, device="cuda")
            
            # 构建齐次坐标
            homo_vertices = torch.cat([vertices_gpu, torch.ones_like(vertices_gpu[:, :1])], dim=1)  # [N, 4]
            
            # 变换到相机坐标系
            cam_coords = homo_vertices @ w2c_gpu.T  # [N, 4]
            
            # 保留点在相机前方(z>0)的信息
            z_positive = cam_coords[:, 2] > 0
            
            # 投影到图像平面
            points_2d = cam_coords[:, :3] @ K_gpu.T  # [N, 3]
            points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # [N, 2] - 透视除法
            
            # 检查点是否在图像范围内
            in_image = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & \
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height) & \
                     z_positive
            
            # 获取相机的alpha mask
            alpha_mask = np.array(camera.image.split()[-1])  # 获取alpha通道
            alpha_mask_gpu = torch.tensor(alpha_mask, dtype=torch.uint8, device="cuda")
            
            # 计算投影点在alpha mask中的坐标 (取整)
            points_2d_int = points_2d.long()
            
            # 只处理在图像内的点
            valid_indices = torch.nonzero(in_image, as_tuple=True)[0]
            
            # 初始化结果掩码 - 所有点默认为可见
            visible_in_camera = torch.ones(vertices_gpu.shape[0], dtype=torch.bool, device="cuda")
            
            if valid_indices.shape[0] > 0:
                valid_points_2d_int = points_2d_int[valid_indices]
                
                # 获取这些点处的alpha值 (考虑坐标系差异，图像坐标原点在左上角)
                alpha_values = alpha_mask_gpu[valid_points_2d_int[:, 1], valid_points_2d_int[:, 0]]
                
                # 判断点是否在前景区域 (alpha > 0)
                not_in_foreground = alpha_values < 0.01
                
             
                visible_in_camera[valid_indices[not_in_foreground]] = False
                
            return in_image,visible_in_camera
        
        def save_simple_tensor_to_ply(tensor, filepath):
            # 将tensor转换为numpy数组
            xyz = tensor.detach().cpu().numpy()
            
            # 定义简化的数据类型
            dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            
            # 创建结构化数组
            elements = np.empty(xyz.shape[0], dtype=dtype)
            elements[:] = list(map(tuple, xyz))
            
            # 创建PlyData对象并写入文件
            from plyfile import PlyElement, PlyData
            vertex_element = PlyElement.describe(elements, 'vertex')
            ply_data = PlyData([vertex_element])
            ply_data.write(filepath)
            
            print(f"已将点云保存到 {filepath}")
        # 在[-1.3, 1.3]范围内生成均匀四面体网格
        scale = 1.3
        grid_size = 64  # 调整网格分辨率
        vertices, tetrahedra = create_tetrahedral_grid(scale, grid_size, grid_size, grid_size)
        
        # 将NumPy数组转换为PyTorch张量并移至GPU
        vertices_gpu = torch.tensor(vertices, dtype=torch.float32, device="cuda")
        tetrahedra_gpu = torch.tensor(tetrahedra, dtype=torch.int64, device="cuda")
        
        # print(f"初始网格: {vertices_gpu.shape[0]}顶点，{tetrahedra_gpu.shape[0]}四面体")
        
        # # 为每个顶点创建标记，初始为False(未在任何相机视角中可见)
        # inside_mask = torch.zeros(vertices_gpu.shape[0], dtype=torch.bool, device="cuda")
        # not_visible_mask = torch.ones(vertices_gpu.shape[0], dtype=torch.bool, device="cuda")
        
        # # 处理每个训练相机
        # for cam in train_cam_infos:
        #     # 获取当前相机中顶点的可见性
        #     in_frustum, not_visible_in_camera = project_vertices_to_camera(vertices_gpu, cam)
            
        #     # 更新总体可见性掩码 - 如果在任一相机中可见则为可见
        #     inside_mask = inside_mask | in_frustum
        #     not_visible_mask = not_visible_mask & not_visible_in_camera
        
        # inside_mask = inside_mask & not_visible_mask
        # print(f"裁剪后保留的顶点数: {inside_mask.sum().item()}/{vertices_gpu.shape[0]}")
        
        # # 检查每个四面体的四个顶点是否都在前景区域
        # tetras_valid = torch.zeros(tetrahedra_gpu.shape[0], dtype=torch.bool, device="cuda")
        # for i in range(4):
        #     tetras_valid = tetras_valid | inside_mask[tetrahedra_gpu[:, i]]
        
        # # 保留至少有一个顶点在前景区域的四面体
        # valid_tetras = tetrahedra_gpu[tetras_valid]
        # print(f"裁剪后保留的四面体数: {valid_tetras.shape[0]}/{tetrahedra_gpu.shape[0]}")
        
        # # 找出仍在使用的顶点
        # used_vertices = torch.zeros(vertices_gpu.shape[0], dtype=torch.bool, device="cuda")
        # for i in range(4):
        #     used_vertices[valid_tetras[:, i]] = True
        
        # # 创建新的顶点索引映射
        # new_indices = torch.zeros(vertices_gpu.shape[0], dtype=torch.int64, device="cuda")
        # new_indices[used_vertices] = torch.arange(used_vertices.sum(), device="cuda")
        
        # # 更新四面体顶点索引
        # remapped_tetras = new_indices[valid_tetras]
        
        # # 提取使用的顶点
        # final_vertices = vertices_gpu[used_vertices].cpu().numpy()
        # final_tetras = remapped_tetras.cpu().numpy()
        
        # print(f"最终网格: {final_vertices.shape[0]}顶点，{final_tetras.shape[0]}四面体")
        mesh = meshio.Mesh(
        points=vertices,
        cells=[("tetra", tetrahedra)]
        )
    
        # 保存为 MSH 文件
        mesh.write(tetra_path, file_format="gmsh") 

        print(f"四面体网格已保存到 {tetra_path}")
    
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
        
    try:
        tetra = fetchmsh(tetra_path)
    except:
        tetra = None

    scene_info = SceneInfo(point_cloud=pcd,
                           tetrahedra=tetra,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}