import torch
import numpy as np
import math
from geo_utils import *

def test_tetrahedron_metrics():
    # 创建测试四面体
    
    # 1. 正四面体（所有边长相等）
    # 正四面体的一种标准表示形式
    a = 1.0
    h = a * math.sqrt(2/3)
    regular_tet = torch.tensor([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*math.sqrt(3)/2, 0],
        [a/2, a*math.sqrt(3)/6, h]
    ], dtype=torch.float32)
    
    # 2. 单位四面体 (000, 100, 010, 001)
    unit_tet = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # 3. 钝角四面体（包含一个钝二面角）
    obtuse_tet = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0.1, 0.1, 0.5]  # 这会创建一个钝角
    ], dtype=torch.float32)
    
    # 创建四面体索引
    tetrahedra = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    
    # 测试所有四面体
    for name, vertices in [("正四面体", regular_tet), 
                          ("单位四面体", unit_tet), 
                          ("钝角四面体", obtuse_tet)]:
        print(f"\n--- {name} ---")
        
        # 计算体积
        volume = compute_tetrahedron_volume(vertices, tetrahedra)[0].item()
        print(f"体积: {volume:.6f}")
        
        # 计算表面积
        surface_area = compute_tetrahedron_surface_area(vertices, tetrahedra)[0].item()
        print(f"表面积: {surface_area:.6f}")
        
        # 计算边长
        edge_lengths = compute_tetrahedron_edge_lengths(vertices, tetrahedra)[0].tolist()
        print(f"边长: {[f'{e:.6f}' for e in edge_lengths]}")
        
        # 计算外接球半径
        circumradius = compute_tetrahedron_circumradius(vertices, tetrahedra)[0].item()
        print(f"外接球半径: {circumradius:.6f}")
        
        # 计算内接球半径
        inradius = compute_tetrahedron_inradius(
            compute_tetrahedron_volume(vertices, tetrahedra), 
            compute_tetrahedron_surface_area(vertices, tetrahedra)
        )[0].item()
        print(f"内接球半径: {inradius:.6f}")
        
        # 计算二面角
        dihedral_angles = compute_tetrahedron_dihedral_angles(vertices, tetrahedra)[0].tolist()
        print(f"二面角(弧度): {[f'{a:.6f}' for a in dihedral_angles]}")
        print(f"二面角(度数): {[f'{math.degrees(a):.6f}' for a in dihedral_angles]}")
        
        # 计算质量度量
        radius_ratio = compute_radius_ratio(vertices, tetrahedra)[0].item()
        print(f"半径比: {radius_ratio:.6f}")
        
        rms_ratio = compute_rms_edge_ratio(vertices, tetrahedra)[0].item()
        print(f"均方根边长比: {rms_ratio:.6f}")
        
        min_angle = compute_min_dihedral_angle(vertices, tetrahedra)[0].item()
        print(f"最小二面角(弧度): {min_angle:.6f}")
        print(f"最小二面角(度数): {math.degrees(min_angle):.6f}")

        min_sine = compute_min_sine(vertices, tetrahedra)[0].item()
        print(f"最小二面角正弦值: {min_sine:.6f}")
        
        condition_number = compute_jacobian_condition_number(vertices, tetrahedra)[0].item()
        print(f"雅可比条件数: {condition_number:.6f}")
        
        # 与理论值比较
        print("\n与理论值比较:")
        if name == "正四面体":
            # 理论值计算 - 正四面体
            theo_volume = math.sqrt(2) * a**3 / 12
            theo_surface_area = math.sqrt(3) * a**2
            theo_circumradius = a * math.sqrt(6) / 4
            theo_inradius = a * math.sqrt(6) / 12
            theo_dihedral = math.acos(1/3)  # 约70.53度
            
            print(f"理论体积: {theo_volume:.6f}, 误差: {abs(volume - theo_volume):.6f}")
            print(f"理论表面积: {theo_surface_area:.6f}, 误差: {abs(surface_area - theo_surface_area):.6f}")
            print(f"理论外接球半径: {theo_circumradius:.6f}, 误差: {abs(circumradius - theo_circumradius):.6f}")
            print(f"理论内接球半径: {theo_inradius:.6f}, 误差: {abs(inradius - theo_inradius):.6f}")
            print(f"理论二面角(弧度): {theo_dihedral:.6f}")
            print(f"理论半径比: 1.000000, 误差: {abs(radius_ratio - 1.0):.6f}")

        elif name == "单位四面体":
            # 理论值计算 - 单位四面体
            theo_volume = 1/6
            theo_surface_area = math.sqrt(2) + math.sqrt(3)
            
            print(f"理论体积: {theo_volume:.6f}, 误差: {abs(volume - theo_volume):.6f}")
            print(f"理论表面积: {theo_surface_area:.6f}, 误差: {abs(surface_area - theo_surface_area):.6f}")

if __name__ == "__main__":
    test_tetrahedron_metrics()