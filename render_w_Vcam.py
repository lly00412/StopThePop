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
from utils.plot_utils import colormap

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, splat_args: ExtendedSettings, render_depth: bool):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    uncert_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "uncert_depth")
    uncert_rgb_path = os.path.join(model_path, name, "ours_{}".format(iteration), "uncert_rgb")
    err_path = os.path.join(model_path, name, "ours_{}".format(iteration), "error")
    mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "common_mask")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(uncert_depth_path, exist_ok=True)
    makedirs(uncert_rgb_path, exist_ok=True)
    makedirs(err_path, exist_ok=True)
    makedirs(mask_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        ###########################
        #  rendering RGB & error
        ###########################
        rendering = render(view, gaussians, pipeline, background, splat_args=splat_args, render_depth=False)["render"]
        gt = view.original_image[0:3, :, :]
        err = torch.mean((rendering - gt)**2,0)
        maxtile = torch.quantile(err.flatten(),0.9)
        topk_err = (err>maxtile)
        err_color = colormap(err,max=maxtile,min=0.)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(err_color, os.path.join(err_path, '{0:05d}'.format(idx) + ".png"))

        ###########################
        #  rendering depth
        ###########################
        depth = render(view, gaussians, pipeline, background, splat_args=splat_args, render_depth=True)["render"][0]
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))

        ###########################
        #  rendering uncertainty
        ###########################
        vir_path = os.path.join(model_path, name, "ours_{}".format(iteration), "vir")
        makedirs(vir_path, exist_ok=True)

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
            torchvision.utils.save_image(vir_depth, os.path.join(vir_path, '{0:05d}'.format(idx) + drt + ".png"))
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

        # depth uncertainty
        vir2rd_depth_sum = vir2rd_depths.sum(0)
        numels = 4. - nv_mask.sum(0)
        vir2rd_depth = torch.zeros_like(rd_depth.squeeze(0))
        vir2rd_depth[numels>0] = vir2rd_depth_sum[numels>0] / numels[numels>0]
        uncert_depth = (rd_depth.squeeze(0) - vir2rd_depth_sum)**2
        bg_mask = (rd_depth.squeeze(0)<1.) # only for nerf synthetic that does not have backgroud
        mask = (rd_depth.squeeze(0)>0.) & bg_mask
        uncert_depth[~mask] = 0.
        depthtile = torch.quantile(uncert_depth.flatten(),0.9)
        topk_depth = (uncert_depth>depthtile)
        uncert_depth_color = colormap(uncert_depth,max=depthtile,min=0.)
        torchvision.utils.save_image(uncert_depth_color, os.path.join(uncert_depth_path, '{0:05d}'.format(idx) + ".png"))

        # rgb uncertainty
        vir2rd_render_sum = vir2rd_renders.sum(0).mean(0,keepdim=True)
        rendering_ = rendering.mean(0,keepdim=True)
        vir2rd_render = torch.zeros_like(rendering_)
        vir2rd_render[numels > 0] = vir2rd_render_sum[numels > 0] / numels[numels > 0]
        uncert_rgb = (rendering_ - vir2rd_render) ** 2
        uncert_rgb[~mask] = 0.
        rendermax = torch.quantile(uncert_rgb.flatten(),0.9)
        topk_rgb = (uncert_rgb>rendermax)
        uncert_rgb_color = colormap(uncert_rgb,max=rendermax,min=0.)
        torchvision.utils.save_image(uncert_rgb_color,
                                     os.path.join(uncert_rgb_path, '{0:05d}'.format(idx) + ".png"))

        ###############################
        # comparing topk
        ###############################
        err_depth = (topk_err & topk_depth).float()
        err_rgb = (topk_err & topk_rgb).float()
        depth_rgb = (topk_depth | topk_rgb).float()
        torchvision.utils.save_image(topk_err.float(),
                                     os.path.join(mask_path, '{0:05d}'.format(idx) + "topk_err" + ".png"))
        torchvision.utils.save_image(err_depth,
                                     os.path.join(mask_path, '{0:05d}'.format(idx) + "err&depth" + ".png"))
        torchvision.utils.save_image(err_rgb,
                                     os.path.join(mask_path, '{0:05d}'.format(idx) + "err&rgb" + ".png"))
        torchvision.utils.save_image(depth_rgb,
                                     os.path.join(mask_path, '{0:05d}'.format(idx) + "depth+rgb" + ".png"))


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