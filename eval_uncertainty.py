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
from scene import Scene
from scene.cameras import VirtualCam
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
import json
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from arguments import ModelParams, PipelineParams, SplattingSettings
from diff_gaussian_rasterization import ExtendedSettings
from gaussian_renderer import GaussianModel
from utils.proj_utils import *
from utils.graphics_utils import getIntrinsicMatrix
from utils.uncert_utils import *


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, splat_args: ExtendedSettings, render_depth: bool):
    plot_path = os.path.join(model_path, name, "ours_{}".format(iteration), "roc")
    makedirs(plot_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        ################################
        #  rendering RGB & error & depth
        ################################
        rendering = render(view, gaussians, pipeline, background, splat_args=splat_args, render_depth=False)["render"]
        depth = render(view, gaussians, pipeline, background, splat_args=splat_args, render_depth=True)["render"][0]
        gt = view.original_image[0:3, :, :]
        err = torch.mean((rendering - gt)**2,0)
        bg_mask = (depth < 1.)  # only for nerf synthetic that does not have backgroud
        mask = (depth > 0.) & bg_mask # (H,W)

        values = {}
        values['error'] = err[mask].flatten()

        ################################
        #  rendering virtual camera
        ################################
        look_at, rd_c2w = extract_scene_center_and_C2W(depth, view)
        rd_c2w = rd_c2w.to(depth.device)
        K = getIntrinsicMatrix(width=view.image_width, height=view.image_height,
                               fovX=view.FoVx, fovY=view.FoVy).to(depth.device) # (4,4)
        GetVcam = VirtualCam(view)
        backwarp = BackwardWarping(out_hw=(view.image_height,view.image_width),
                                   device=depth.device,K= K)
        rd_depth = depth.clone().unsqueeze(0).unsqueeze(0)
        vir_depths = []
        vir_renders = []
        rd2virs = []
        for drt in ['u','d','l','r']:
            vir_view = GetVcam.get_near_cam_by_look_at(look_at=look_at,direction=drt)
            vir_render = render(vir_view, gaussians, pipeline, background, splat_args=splat_args, render_depth=False)[
                "render"]
            vir_depth = render(vir_view, gaussians, pipeline, background, splat_args=splat_args, render_depth=True)[
                "render"][0]
            vir_w2c = vir_view.world_view_transform.transpose(0,1)
            rd2vir = vir_w2c @ rd_c2w
            rd2virs.append(rd2vir)
            vir_depths.append(vir_depth.unsqueeze(0))
            vir_renders.append(vir_render)
        rd_depths = rd_depth.repeat(4,1,1,1)
        rd_renders = rendering.unsqueeze(0).repeat(4,1,1,1)
        vir_depths = torch.stack(vir_depths)
        rd2virs = torch.stack(rd2virs)
        vir2rd_renders, vir2rd_depths, nv_mask = backwarp(img_src=rd_renders,depth_src=vir_depths,depth_tgt=rd_depths,tgt2src_transform=rd2virs)

        ################################
        #  compute uncertainty by l2 diff
        ################################
        # depth uncertainty
        vir2rd_depth_sum = vir2rd_depths.sum(0)
        numels = 4. - nv_mask.sum(0)
        vir2rd_depth = torch.zeros_like(rd_depth.squeeze(0))
        vir2rd_depth[numels>0] = vir2rd_depth_sum[numels>0] / numels[numels>0]
        depth_l2 = (rd_depth.squeeze(0) - vir2rd_depth_sum)**2
        depth_l2 = depth_l2.squeeze(0)
        values['depth_l2'] = depth_l2[mask].flatten()

        # rgb uncertainty
        vir2rd_render_sum = vir2rd_renders.sum(0).mean(0,keepdim=True)
        rendering_ = rendering.mean(0,keepdim=True)
        vir2rd_render = torch.zeros_like(rendering_)
        vir2rd_render[numels > 0] = vir2rd_render_sum[numels > 0] / numels[numels > 0]
        rgb_l2 = (rendering_ - vir2rd_render) ** 2
        rgb_l2 = rgb_l2.squeeze(0)
        values['rgb_l2'] = rgb_l2[mask].flatten()

        ################################
        #  compute uncertainty by variance
        ################################
        vis_mask = (nv_mask.sum(0)<1.)
        # depth uncertainty
        depths = torch.cat([vir2rd_depths, rd_depth])
        depth_var = depths.var(0)
        depth_var[~vis_mask] = 0.
        depth_var = depth_var.squeeze(0)
        values['depth_var'] = depth_var[mask].flatten()

        # rgb uncertainty
        renderings = torch.cat([vir2rd_renders, rendering.unsqueeze(0)])
        render_var = renderings.var(0).mean(0, keepdim=True)
        render_var[~vis_mask] = 0.
        render_var = render_var.squeeze(0)
        values['rgb_var'] = render_var[mask].flatten()

        ################################
        #  compute roc and auc
        ################################
        ROCs = {}
        AUCs = {}
        opt_label = 'error'
        for val in values.keys():
            roc,auc = compute_roc(opt=values[opt_label],est=values[val], intervals=10)
            ROCs[val] = roc.cpu().numpy()
            AUCs[val] = auc

        plot_file = os.path.join(plot_path, '{0:05d}'.format(idx) + ".png")
        txt_file = os.path.join(plot_path, '{0:05d}'.format(idx) + ".txt")
        plot_roc(ROC_dict=ROCs, fig_name=plot_file, opt_label='error')
        write_auc(AUC_dict=AUCs,txt_name=txt_file)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, splat_args: ExtendedSettings, render_depth: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_test=args.skip_test, skip_train=args.skip_train)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, splat_args, render_depth)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, splat_args, render_depth)
        
        # write number of gaussians too
        num_gaussians = scene.gaussians.get_xyz.shape[0]
        with open(os.path.join(dataset.model_path, "point_cloud", f'iteration_{scene.loaded_iter}', 'num_gaussians.json'), 'w') as fp:
            json.dump(obj={
                "num_gaussians": num_gaussians,
            }, fp=fp, indent=2)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    ss = SplattingSettings(parser, render=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    splat_args = ss.get_settings(args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, splat_args, args.render_depth)