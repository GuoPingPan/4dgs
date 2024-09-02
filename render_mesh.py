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
from utils.render_utils import generate_path, generate_path_fixed_time
from utils.data_utils import CameraDataset
from utils.mesh_utils import GaussianExtractor
import open3d as o3d
import numpy as np

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, gaussian_dim=4, rot_4d=True)
        scene = Scene(dataset, gaussians, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 原来生成mesh应该是在静态场景下，用近 200 个视角生成 mesh
        # 现在 4dgs 在同一时刻只有 20 个视角，所以插值出相同时刻的 200 个视角
        # 将时间给他叠加
        
        num_step = 20
        timestep = 10 / 20
        cam_all = scene.train_cameras[1].copy()  # List[Camera]
        cam_select = [cam_all[i * 300].cuda() for i in range(20)]
        name = "dynamic"
        mesh_path = os.path.join(dataset.model_path, name, "ours_{}".format(iteration), "mesh")
        for i in range(num_step):
            torch.cuda.empty_cache()
            
            # n_fames = 80
            # cam_traj = generate_path_fixed_time(cam_select, n_frames=n_fames, time_step=timestep * i)
            
            gaussExtractor = GaussianExtractor(gaussians, render, pipeline, bg_color=background)

            print("export mesh ...")
            os.makedirs(mesh_path, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0
            # gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.reconstruction(cam_select)
            # extract the mesh and save
            if args.unbounded:
                name = f'fuse_unbounded_{i}.ply'
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                name = f'fuse_{i}.ply'
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            
            o3d.io.write_triangle_mesh(os.path.join(mesh_path, name), mesh)
            print("mesh saved at {}".format(os.path.join(mesh_path, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(mesh_path, name.replace('.ply', '_post.ply')), mesh_post)
    
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)