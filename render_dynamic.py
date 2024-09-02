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
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.render_utils import generate_path, create_video_from_images, create_video_from_images_h264
from utils.data_utils import CameraDataset
import sys

def do_system(arg):
    print(f"==== running: {arg}")
    import ipdb; ipdb.set_trace()
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path = os.path.join(base_path, "renders")
    gts_path = os.path.join(base_path, "gt")
    depth_path = os.path.join(base_path, "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    depth_images = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view[1].cuda(), gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth = render_pkg["depth"]
        depth_images.append(depth) 
        # gt = view[0][0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    
    
    import matplotlib.pyplot as plt
    import time
    
    cmap = plt.get_cmap('magma')
    t1 = time.time()
    depths_max = float(max([torch.max(depth) for depth in depth_images]))
    print(f"Time to get max depth: {time.time() - t1}")
    
    t1 = time.time()
    depths_norm = [d / depths_max for d in depth_images]
    print(f"Time to normalize depths: {time.time() - t1}")
    
    t1 = time.time()
    for i, depth_image in enumerate(tqdm(depths_norm)):
        depth_image = depth_image.squeeze().cpu().numpy()
        plt.imsave(os.path.join(depth_path, '{0:05d}'.format(i) + ".png"), cmap(depth_image))  # 把函数作用写在 [ for ] 里面很慢
    print(f"Time to save images: {time.time() - t1}")
    
    
    # create_video_from_images_h264(base_path, output_name="video", fps=40)
    do_system(f"ffmpeg -y -r 40 -i {render_path}/%05d.png -c:v libx264 -vf fps=40 -pix_fmt yuv420p {base_path}/renders.mp4")
    do_system(f"ffmpeg -y -r 40 -i {depth_path}/%05d.png -c:v libx264 -vf fps=40 -pix_fmt yuv420p {base_path}/depth.mp4")


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        
        
        # interpolate between two cameras
        n_fames = 2000

        cam_all = scene.train_cameras[1].copy()  # List[Camera]
        cam_select = [cam_all[i * 300] for i in range(20)]
        
        cam_traj = generate_path(cam_select, n_frames=n_fames)
        cam_traj = [(0, cam_traj[i]) for i in range(len(cam_traj))]  # List[Tuple[threshold, Camera]]
        
        render_set(dataset.model_path, "dynamic", iteration, cam_traj, gaussians, pipeline, background)
        
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)