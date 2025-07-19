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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui, render_tet, render_htet
import sys
from scene import Scene, GaussianModel, TetScene, TetrahedraModel, HierarchicalTetScene, HierarchicalTetrahedraModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
import torchvision
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
 
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # gaussians = GaussianModel(dataset.sh_degree)
    # scene = Scene(dataset, gaussians)

    tets = HierarchicalTetrahedraModel(dataset.sh_degree,use_cell_opacity=opt.use_cell_opacity,use_cell_scale=opt.use_cell_scale,max_depth=opt.max_depth, optimizable_rotation=False)
    scene = HierarchicalTetScene(dataset, tets)
    tets.setup_model(pipe)
    
    tets.training_setup(opt)
    
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)  
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_rgb_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
     
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, tets, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        tets.update_learning_rate(iteration)
                
        alpha_ratio =  1.0 # max(min(iteration / opt.max_pe_iter, 1.0), 0.0)
        tets.alpha_ratio = alpha_ratio  

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            tets.oneupSHdegree()

        # if (iteration % opt.opacity_reset_interval == 500 and iteration > 3000 and tets.freeze_inn):
        #     tets.unfreeze_inn()

        if pipe.freeze:    
            tets.freeze_inn()
        if iteration == 1:
            tets.freeze_inn()
        # if iteration == 0:
        #     tets.freeze_inn()
        # if iteration == opt.prune_only_iter:
        #     tets.unfreeze_inn()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render_htet(viewpoint_cam, tets, bg, lod=tets.current_lod_depth, feature_interp_mode=opt.feature_interp_mode)
        image, viewspace_point_tensor, visibility_filter, radii, mask = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_alpha"]
        

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if opt.lambda_mask > 0.0:
            gt_mask = viewpoint_cam.alpha_mask.cuda()
            Lmask = opt.lambda_mask * l2_loss(mask, gt_mask)
        else:
            Lmask = 0.0
        
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        reg_loss = Lmask
        if opt.lambda_relu_tet > 0.0:
            reg_loss += opt.lambda_relu_tet * tets.signed_volume_tet_loss()
        if opt.lambda_quality > 0.0:
            if iteration > opt.lambda_quality_final_iter:
                quality_loss = opt.lambda_quality_final * tets.compute_quality_loss(threshold=opt.quality_threshold)
            else:
                quality_loss = opt.lambda_quality * tets.compute_quality_loss(threshold=opt.quality_threshold)
        else:
            quality_loss = torch.tensor(0.0, device="cuda")
        # reg_loss2 = -tets.signed_volume_of_tetra().sum()
        # reg_loss2 = torch.relu(-tets.signed_volume_of_tetra()).sum()
        # reg_loss3 = tets.edge_ratio_loss()
        loss = rgb_loss  +  reg_loss + quality_loss
        loss.backward()

        if not tets.freeze:
            # 手动将 INN 网络和 deformation_code 的 NaN/inf 梯度设置为 0
            if tets.inn_network is not None:
                # 处理所有参数的梯度中的NaN和inf
                for param_group in tets.optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0, out=param.grad)
                
            # 添加 INN 网络梯度裁剪
            if tets.inn_network is not None and tets.clip_grad_norm_inn:
                torch.nn.utils.clip_grad_norm_(tets.inn_network.parameters(), tets.max_grad_norm_inn)
            
            # 如果 deformation_code 也被视为 INN 网络的一部分并且需要裁剪
            if hasattr(tets, 'deformation_code') and tets.deformation_code is not None and tets.deformation_code.grad is not None and tets.clip_grad_norm_inn:
                torch.nn.utils.clip_grad_norm_([tets.deformation_code], tets.max_grad_norm_inn)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_rgb_loss_for_log = 0.4 * rgb_loss.item() + 0.6 * ema_rgb_loss_for_log
            if iteration % 10 == 0:
                num_signed = (tets.signed_volume_tet() < 0).sum().item()
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","RGB_Loss": f"{ema_rgb_loss_for_log:.{7}f}", "Quality_Loss": f"{quality_loss.item():.7f}", "Num Signed": num_signed})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, rgb_loss, reg_loss, l1_loss, quality_loss, iter_start.elapsed_time(iter_end), testing_iterations, opt.feature_interp_mode, scene, render_htet, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                tets.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # if iteration > opt.prune_only_iter:
                #     tets.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # if iteration < opt.prune_only_iter and iteration % opt.densification_interval == 0 and iteration > opt.prune_from_iter:
                #                         # 渲染当前视角的图像（密度化前）
                #                        # 在密度化前保存当前视角的渲染图像
                #     if not os.path.exists(os.path.join(scene.model_path, "densify_comparison")):
                #         os.makedirs(os.path.join(scene.model_path, "densify_comparison"), exist_ok=True)
                        
                   
                #     with torch.no_grad():
                #         pre_densify_render = render_htet(viewpoint_cam, tets, bg, lod=tets.current_lod_depth, feature_interp_mode=opt.feature_interp_mode)
                #         pre_densify_image = torch.clamp(pre_densify_render["render"], 0.0, 1.0)
                #         torchvision.utils.save_image(pre_densify_image, os.path.join(scene.model_path, "densify_comparison", f"iter_{iteration}_pre_densify.png"))
                    
                    
                #     tets.prune_only(0.05)


                #     # 渲染同一视角的图像（密度化后）
                #     with torch.no_grad():
                #         post_densify_render = render_htet(viewpoint_cam, tets, bg, lod=tets.current_lod_depth, feature_interp_mode=opt.feature_interp_mode)
                #         post_densify_image = torch.clamp(post_densify_render["render"], 0.0, 1.0)
                #         torchvision.utils.save_image(post_densify_image, os.path.join(scene.model_path, "densify_comparison", f"iter_{iteration}_post_densify.png"))
                        
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #     size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                #     # 在密度化前保存当前视角的渲染图像
                #     if not os.path.exists(os.path.join(scene.model_path, "densify_comparison")):
                #         os.makedirs(os.path.join(scene.model_path, "densify_comparison"), exist_ok=True)
                        
                #     # 渲染当前视角的图像（密度化前）
                #     with torch.no_grad():
                #         pre_densify_render = render_htet(viewpoint_cam, tets, bg, lod=tets.current_lod_depth, feature_interp_mode=opt.feature_interp_mode)
                #         pre_densify_image = torch.clamp(pre_densify_render["render"], 0.0, 1.0)
                #         torchvision.utils.save_image(pre_densify_image, os.path.join(scene.model_path, "densify_comparison", f"iter_{iteration}_pre_densify.png"))
                    
                #     # 执行密度化和剪枝操作
                #     tets.densify_and_prune(viewpoint_cam,opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    
                #     # 渲染同一视角的图像（密度化后）
                #     with torch.no_grad():
                #         post_densify_render = render_htet(viewpoint_cam, tets, bg, lod=tets.current_lod_depth, feature_interp_mode=opt.feature_interp_mode)
                #         post_densify_image = torch.clamp(post_densify_render["render"], 0.0, 1.0)
                #         torchvision.utils.save_image(post_densify_image, os.path.join(scene.model_path, "densify_comparison", f"iter_{iteration}_post_densify.png"))
                        
                #         # 保存原始图像作为参考
                #         gt_image = torch.clamp(viewpoint_cam.original_image.cuda(), 0.0, 1.0)
                #         torchvision.utils.save_image(gt_image, os.path.join(scene.model_path, "densify_comparison", f"iter_{iteration}_gt.png"))
                        
                #         print(f"\n[ITER {iteration}] 已保存密度化前后的对比图像")
                    
                #     # 重置摄像机视点栈，确保下一次迭代使用新的随机视点
                #     viewpoint_stack = scene.getTrainCameras().copy()
                    
                # if (iteration % opt.opacity_reset_interval == 0 and iteration > opt.densify_from_iter) or (dataset.white_background and iteration == opt.densify_from_iter):
                #     tets.reset_opacity()
                    

            # Optimizer step
            if iteration < opt.iterations:
                tets.optimizer.step()
                tets.optimizer.zero_grad(set_to_none = True)

            # if iteration < opt.densify_until_iter:
            #     if (iteration % opt.opacity_reset_interval == 0 and iteration > opt.densify_from_iter) or (dataset.white_background and iteration == opt.densify_from_iter):
            #         tets.freeze_inn()
                    
            if (iteration % 3000 == 0):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((tets.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    
    # 训练结束后，计算并返回最终的PSNR和L1指标
    torch.cuda.empty_cache()
    final_metrics = {}
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},)

    
   # 训练结束后，计算并返回最终的PSNR和L1指标
    torch.cuda.empty_cache()
    final_metrics = {}
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},)
    with torch.no_grad():
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0: 
                l1_test = 0.0
                psnr_test = 0.0
                
                # 使用更小的批次处理相机
                batch_size = 5  # 可以根据显存情况调整这个值
                test_cameras = config['cameras']
                num_cameras = len(test_cameras)
                
                for batch_start in range(0, num_cameras, batch_size):
                    batch_end = min(batch_start + batch_size, num_cameras)
                    batch_cameras = test_cameras[batch_start:batch_end]
                    
                    # 使用torch.no_grad()避免存储计算图
                    
                    for viewpoint in batch_cameras:
                        image = torch.clamp(render_htet(viewpoint, scene.tets, bg_color=background, lod=scene.tets.current_lod_depth, feature_interp_mode=opt.feature_interp_mode)["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        
                        # 确保删除临时变量
                        del image, gt_image
                
                    # 每个批次后清理缓存
                    torch.cuda.empty_cache()
                    print(f"\n[PROGRESS] 已处理 {batch_end}/{num_cameras} 个测试视角")
                    
                psnr_test /= num_cameras
                l1_test /= num_cameras
                final_metrics[f"{config['name']}_psnr"] = psnr_test.item()
                final_metrics[f"{config['name']}_l1_loss"] = l1_test.item()
                print(f"\n[FINAL] {config['name']}: L1 {l1_test} PSNR {psnr_test}")
        
    return final_metrics
    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, reg_loss, l1_loss, quality_loss, elapsed, testing_iterations, feature_interp_mode, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/quality_loss', quality_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    image = torch.clamp(renderFunc(viewpoint, scene.tets, bg_color=renderArgs[1], lod=scene.tets.current_lod_depth, feature_interp_mode=feature_interp_mode)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        # if tb_writer:
        #     tb_writer.add_histogram("scene/opacity_histogram", scene.tets._opacity, iteration)
        #     tb_writer.add_scalar('total_points', scene.tets.get_xyz().shape[0], iteration)
        torch.cuda.empty_cache()

def render_set(model_path, name, iteration, views, tets, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    import os
    import torchvision
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, tets, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def main_train(args=None):

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # 如果没有提供参数，则解析命令行参数
    if args is None:
        # Set up command line argument parser

        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6229)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 50_000])
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7000,  30_000, 50_000])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        parser.add_argument("--normalization", action='store_true', default=False)
        parser.add_argument("--hidden_size", nargs="+", type=int, default=[128,128])
        parser.add_argument("--using_random_init", action='store_true', default=False)
        parser.add_argument("--freeze", action='store_true', default=False)
        
        args = parser.parse_args(sys.argv[1:])

    args.save_iterations.append(args.iterations)
    lp.init_mesh = args.using_random_init
    print("Optimizing " + args.model_path)
    pp.normalization = args.normalization
    pp.hidden_size = args.hidden_size
    pp.freeze = args.freeze
    
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Enable anomaly detection for autograd if requested
    if args.detect_anomaly:
        print("Enabling anomaly detection for autograd.")
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.autograd.set_detect_anomaly(False) # Explicitly disable if not set, or rely on PyTorch default

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly) # This line can be removed if the above block is used
    metrics = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
    return metrics

if __name__ == "__main__":
    main_train()
