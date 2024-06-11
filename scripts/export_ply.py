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
    # params = np.load("/mnt/massive/skr/SplaTAM/experiments/Apartment/Post_SplaTAM_Opt/params.npz", allow_pickle=True)
    params = np.load("/mnt/massive/skr/SplaTAM/pcd_save/room0_0/update_params_0.npz", allow_pickle=True)
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']

    # # draw initial trajectory
    # Trajectory = {} 
    # trajectory = []
    # for i in range(config['data']['num_frames']):
    #     xyz = params['cam_trans'][..., i].squeeze(0)
    #     quat = params['cam_unnorm_rots'][..., i].squeeze(0)
    #     pose = np.concatenate((xyz, quat), axis=0)
    #     trajectory.append(pose)
    # trajectory = np.array(trajectory)
    # trajectory = torch.from_numpy(trajectory)
    # trajectory = w2c_to_c2w(trajectory)
    # Trajectory['xyz'] = np.ones((config['data']['num_frames'], 3))
    # Trajectory['scales'] = np.ones((config['data']['num_frames'], scales.shape[1]))
    # Trajectory['rotations'] = np.ones((config['data']['num_frames'], rotations.shape[1]))
    # Trajectory['rgbs'] = np.ones((config['data']['num_frames'], rgbs.shape[1]))
    # Trajectory['opacities'] = np.ones((config['data']['num_frames'], opacities.shape[1]))
    # for i in range(config['data']['num_frames']):
    #     Trajectory['xyz'][i] = trajectory[i, :3]
    #     Trajectory['scales'][i] = scales[0] +1
    #     Trajectory['rotations'][i] = [1, 0, 0, 0]
    #     Trajectory['rgbs'][i] = [0, 0, 1]
    #     Trajectory['opacities'][i] = [1]
    
    # draw Lnet finer trajectory
    Lnet_est_traj= {} 
    Lnet_est_traj['xyz'] = np.ones((config['data']['num_frames'], 3))
    Lnet_est_traj['scales'] = np.ones((config['data']['num_frames'], scales.shape[1]))
    Lnet_est_traj['rotations'] = np.ones((config['data']['num_frames'], rotations.shape[1]))
    Lnet_est_traj['rgbs'] = np.ones((config['data']['num_frames'], rgbs.shape[1]))
    Lnet_est_traj['opacities'] = np.ones((config['data']['num_frames'], opacities.shape[1]))
    # Lnet_pose_est = np.load(f"./pcd_save/{config['run_name']}/reference_trans_pose_2519.npy")
    Lnet_pose_est = np.load(f"./pcd_save/{config['run_name']}/reference_GSupdate_Lnet_est_pose_00.npy")
    for i in range(config['data']['num_frames']):
        Lnet_est_traj['xyz'][i] = Lnet_pose_est[i, :3]
        Lnet_est_traj['scales'][i] = scales[0] +1
        Lnet_est_traj['rotations'][i] = [1, 0, 0, 0]
        Lnet_est_traj['rgbs'][i] = [0, 1, 0]
        Lnet_est_traj['opacities'][i] = [1]

    # num_frame = 12595
    # # draw gt trajectory
    # initial_traj= {} 
    # initial_traj['xyz'] = np.ones((num_frame, 3))
    # initial_traj['scales'] = np.ones((num_frame, scales.shape[1]))
    # initial_traj['rotations'] = np.ones((num_frame, rotations.shape[1]))
    # initial_traj['rgbs'] = np.ones((num_frame, rgbs.shape[1]))
    # initial_traj['opacities'] = np.ones((num_frame, opacities.shape[1]))

    # init_pose = np.loadtxt("/mnt/massive/skr/SplaTAM/data/Apartment/estimate_c2w_list.txt")
    # # init_pose = np.loadtxt("/mnt/massive/skr/SplaTAM/data/Replica/room0/traj.txt")
    # init_pose = init_pose.reshape(-1, 4, 4)

    # for i in range(num_frame):
    #     initial_traj['xyz'][i] = init_pose[i, :3, 3]
    #     initial_traj['scales'][i] = scales[0] +1
    #     initial_traj['rotations'][i] = [1, 0, 0, 0]
    #     initial_traj['rgbs'][i] = [0, 0, 1]
    #     initial_traj['opacities'][i] = [1]


    # draw gt trajectory
    gt_traj= {} 
    gt_traj['xyz'] = np.ones((config['data']['num_frames'], 3))
    gt_traj['scales'] = np.ones((config['data']['num_frames'], scales.shape[1]))
    gt_traj['rotations'] = np.ones((config['data']['num_frames'], rotations.shape[1]))
    gt_traj['rgbs'] = np.ones((config['data']['num_frames'], rgbs.shape[1]))
    gt_traj['opacities'] = np.ones((config['data']['num_frames'], opacities.shape[1]))

    # gt_pose = np.loadtxt("/mnt/massive/skr/SplaTAM/data/Apartment/gt_c2w_list.txt")
    gt_pose = np.loadtxt("/mnt/massive/skr/SplaTAM/pcd_save/room0_0/gt_all_frames.txt")
    gt_pose = gt_pose.reshape(-1, 4, 4)

    for i in range(config['data']['num_frames']):
        gt_traj['xyz'][i] = gt_pose[i, :3, 3]
        gt_traj['scales'][i] = scales[0] +1
        gt_traj['rotations'][i] = [1, 0, 0, 0]
        gt_traj['rgbs'][i] = [1, 0, 0]
        gt_traj['opacities'][i] = [1]

    # means_xyz = np.concatenate((initial_traj['xyz'], Lnet_est_traj['xyz'], gt_traj['xyz']), axis=0)
    # scales_combined = np.concatenate((initial_traj['scales'], Lnet_est_traj['scales'], gt_traj['scales']), axis=0)
    # rotations_combined = np.concatenate((initial_traj['rotations'], Lnet_est_traj['rotations'], gt_traj['rotations']), axis=0)
    # rgbs_combined = np.concatenate((initial_traj['rgbs'], Lnet_est_traj['rgbs'], gt_traj['rgbs']), axis=0)
    # opacities_combined = np.concatenate((initial_traj['opacities'], Lnet_est_traj['opacities'], gt_traj['opacities']), axis=0)

    means_xyz = np.concatenate((means, Lnet_est_traj['xyz'], gt_traj['xyz']), axis=0)
    scales_combined = np.concatenate((scales, Lnet_est_traj['scales'], gt_traj['scales']), axis=0)
    rotations_combined = np.concatenate((rotations, Lnet_est_traj['rotations'], gt_traj['rotations']), axis=0)
    rgbs_combined = np.concatenate((rgbs, Lnet_est_traj['rgbs'], gt_traj['rgbs']), axis=0)
    opacities_combined = np.concatenate((opacities, Lnet_est_traj['opacities'], gt_traj['opacities']), axis=0)

    ply_path = os.path.join('./pcd_save', config['run_name'], "splat.ply")
    save_ply(ply_path, means_xyz, scales_combined, rotations_combined, rgbs_combined, opacities_combined)
    # save_ply(ply_path, means, scales, rotations, rgbs, opacities)
