import os
import argparse
import torch
from importlib.machinery import SourceFileLoader

import numpy as np
from plyfile import PlyData, PlyElement
from init_pose import w2c_to_c2w

# Spherical harmonic constant
C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = config['run_name']
    
    # params_path = os.path.join(work_path, run_name, "params.npz")
    # params = dict(np.load(params_path, allow_pickle=True))
    # params = np.load("/mnt/massive/skr/SplaTAM/pcd_save/apartment_0/params_2519update.npz", allow_pickle=True)
    params = np.load("/mnt/massive/skr/SplaTAM/experiments/Apartment/Post_SplaTAM_Opt/params.npz", allow_pickle=True)
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']

    Trajectory = {} 
    trajectory = []
    for i in range(config['data']['num_frames']):
        xyz = params['cam_trans'][..., i].squeeze(0)
        quat = params['cam_unnorm_rots'][..., i].squeeze(0)
        pose = np.concatenate((xyz, quat), axis=0)
        trajectory.append(pose)
    trajectory = np.array(trajectory)
    trajectory = torch.from_numpy(trajectory)
    trajectory = w2c_to_c2w(trajectory)
    Trajectory['xyz'] = np.ones((config['data']['num_frames'], 3))
    Trajectory['scales'] = np.ones((config['data']['num_frames'], scales.shape[1]))
    Trajectory['rotations'] = np.ones((config['data']['num_frames'], rotations.shape[1]))
    Trajectory['rgbs'] = np.ones((config['data']['num_frames'], rgbs.shape[1]))
    Trajectory['opacities'] = np.ones((config['data']['num_frames'], opacities.shape[1]))
    for i in range(config['data']['num_frames']):
        Trajectory['xyz'][i] = trajectory[i, :3]
        Trajectory['scales'][i] = scales[0] +1
        Trajectory['rotations'][i] = [1, 0, 0, 0]
        Trajectory['rgbs'][i] = [1, 0, 0]
        Trajectory['opacities'][i] = [1]
    
    Lnet_est_traj= {} 
    Lnet_est_traj['xyz'] = np.ones((config['data']['num_frames'], 3))
    Lnet_est_traj['scales'] = np.ones((config['data']['num_frames'], scales.shape[1]))
    Lnet_est_traj['rotations'] = np.ones((config['data']['num_frames'], rotations.shape[1]))
    Lnet_est_traj['rgbs'] = np.ones((config['data']['num_frames'], rgbs.shape[1]))
    Lnet_est_traj['opacities'] = np.ones((config['data']['num_frames'], opacities.shape[1]))
    Lnet_pose_est = np.load(f"./pcd_save/{config['run_name']}/c2w_Lnet_pose_est_2519.npy")
    # Lnet_pose_est = np.load(f"./pcd_save/Lnet_pose_est.npy")
    for i in range(config['data']['num_frames']):
        Lnet_est_traj['xyz'][i] = Lnet_pose_est[i, :3]
        Lnet_est_traj['scales'][i] = scales[0] +1
        Lnet_est_traj['rotations'][i] = [1, 0, 0, 0]
        Lnet_est_traj['rgbs'][i] = [0, 0, 1]
        Lnet_est_traj['opacities'][i] = [1]

    # means_xyz = np.concatenate((means, Trajectory['xyz']), axis=0)
    # scales_combined = np.concatenate((scales, Trajectory['scales']), axis=0)
    # rotations_combined = np.concatenate((rotations, Trajectory['rotations']), axis=0)
    # rgbs_combined = np.concatenate((rgbs, Trajectory['rgbs']), axis=0)
    # opacities_combined = np.concatenate((opacities, Trajectory['opacities']), axis=0)

    means_xyz = np.concatenate((means, Trajectory['xyz'], Lnet_est_traj['xyz']), axis=0)
    scales_combined = np.concatenate((scales, Trajectory['scales'], Lnet_est_traj['scales']), axis=0)
    rotations_combined = np.concatenate((rotations, Trajectory['rotations'], Lnet_est_traj['rotations']), axis=0)
    rgbs_combined = np.concatenate((rgbs, Trajectory['rgbs'], Lnet_est_traj['rgbs']), axis=0)
    opacities_combined = np.concatenate((opacities, Trajectory['opacities'], Lnet_est_traj['opacities']), axis=0)

    ply_path = os.path.join('./pcd_save', config['run_name'], "splat.ply")
    save_ply(ply_path, means_xyz, scales_combined, rotations_combined, rgbs_combined, opacities_combined)
    # save_ply(ply_path, means, scales, rotations, rgbs, opacities)
