import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)
print("System Paths:")
for p in sys.path:
    print(p)

import numpy as np
from tqdm import tqdm
import itertools
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from utils.slam_external import build_rotation
from utils.keyframe_selection import keyframe_selection_overlap
from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset, ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset, ScannetPPDataset, NeRFCaptureDataset)

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def cos_sim(idx, alpha_list):
    cos_sim_list = []
    for i in range(len(alpha_list)):
        vec = alpha_list[idx] @ alpha_list[i]
        norm = torch.norm(alpha_list[idx]) * torch.norm(alpha_list[i])
        cos_sim = vec / norm
        cos_sim_list.append(cos_sim.cpu().numpy())
    return cos_sim_list

def pose_distance(pose, init_pose):
    distance = []
    d = [0, 0, 0]
    group_distance = []
    for i in range(len(init_pose)):
        for j in range(3):
            d[j] = pose[j] - init_pose[i][j]
        # calculate distance between pose and init_pose[i]
        distance = np.linalg.norm(d)
        group_distance.append(distance)
    return group_distance

def distance_cal(pose, init_pose):
    d = []
    for i in range(3):
        d.append(pose[i] - init_pose[i])
    L2_distance = torch.sum(torch.pow(torch.tensor(d), 2))

    rot = build_rotation(F.normalize(pose[3:].reshape(1, -1))).squeeze(0) # 3*3
    init_rot = build_rotation(F.normalize(init_pose[3:].reshape(1, -1))).squeeze(0) # 3*3

    # R = torch.mm(rot.transpose(0, 1), init_rot)
    # t = torch.trace(R)
    # angle = torch.acos((t - 1) / 2)

    # X = torch.tensor([1, 0, 0]).float().to('cuda')
    # alpha_pose = X @ rot
    # alpha_init_pose = X @ init_rot
    # vec = alpha_pose @ alpha_init_pose
    # norm = torch.norm(alpha_pose) * torch.norm(alpha_init_pose)
    # cos_sim = vec / norm

    distance = L2_distance
    # distance = L2_distance
    return distance

def group_matrix_generate(params, num_frames=100):
    transfer=MinMaxScaler(feature_range=(0,1)) 
    # get each frame's pose
    init_pose = []
    init_quat = []
    for idx in range(num_frames):
        init_pose.append(params['cam_trans'][..., idx].detach().cpu().numpy())
        init_quat.append(params['cam_unnorm_rots'][..., idx].detach().cpu().numpy())
    
    init_pose = np.concatenate(init_pose, axis=0)

    # calculate standard angle of each pose
    init_quat = np.concatenate(init_quat)
    init_quat = torch.from_numpy(init_quat).to('cuda')
    init_rot = build_rotation(F.normalize(init_quat)) # G*3*3
    alpha_list = []
    X = torch.tensor([1, 0, 0]).float().to('cuda')
    for i in range(init_rot.shape[0]):
        alpha = X @ init_rot[i]
        alpha_list.append(alpha)

    # calculate L2 distance of each pose
    group_matrix = []
    for idx, pose in enumerate(tqdm(init_pose, colour='CYAN')):
        # calculate pose_distance
        pose_distance_list = pose_distance(pose, init_pose)
        pose_distance_list = np.array(pose_distance_list)
        pose_distance_list = transfer.fit_transform(pose_distance_list.reshape(-1, 1)).flatten() # normalize

        # calculate angle_distance
        cos_sim_list = cos_sim(idx, alpha_list)
        cos_sim_list = np.array(cos_sim_list)

        # calculate weighted_distance
        weighted_distance = 0.5 * pose_distance_list + 0.5 * (1 - cos_sim_list)
        weighted_distance = [(i, v) for i, v in enumerate(weighted_distance)]
        weighted_distance = np.array(weighted_distance)

        weighted_distance = weighted_distance[weighted_distance[:, 1].argsort()] # sort by distance
        group_matrix.append([x[0] for x in weighted_distance[:20]])    # select 20 nearest pose

    group_matrix = np.array(group_matrix)
    # print(group_matrix)
    # print(group_matrix.shape)
    # distances = np.array(distances)

    return group_matrix

# params = np.load('/mnt/massive/skr/SplaTAM/experiments/Replica/room0_0/params.npz')
# group_matrix_generate(params)


# generate group matrix with keyframe_overlape
def group_matrix_generate_with_overlap(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    if "gaussian_distribution" not in config:
        config['gaussian_distribution'] = "isotropic"
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    _, _, intrinsics, _ = dataset[0]
    intrinsics = intrinsics[:3, :3]

    # params = dict(np.load("/mnt/massive/skr/SplaTAM/experiments/Replica/room0_0/params2000.npz", allow_pickle=True))
    params = dict(np.load("/mnt/massive/skr/SplaTAM/experiments/Apartment/Post_SplaTAM_Opt/params.npz", allow_pickle=True))
    params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
    keyframe_list = []
    for time_idx in tqdm(range(num_frames)):
        # Get the current estimated rotation & translation
        curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        # Initialize Keyframe Info
        curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c}
        # Add to keyframe list
        keyframe_list.append(curr_keyframe)
    print(len(keyframe_list))
    Group_list = []
    for time_idx in tqdm(range(num_frames)):
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx]
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        num_keyframes = 20
        curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4).cuda().float()
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list, num_keyframes)
        if time_idx != selected_keyframes[0]:
            selected_keyframes = [time_idx] + [k for k in selected_keyframes if k != time_idx]
        print(selected_keyframes)
        Group_list.append(selected_keyframes[:20])

    # Create directory if not exists
    if not os.path.exists(f'./pcd_save/{config["run_name"]}'):
        os.makedirs(f'./pcd_save/{config["run_name"]}')
    np.save(f'./pcd_save/{config["run_name"]}/Group_list.npy', Group_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # # Set Experiment Seed
    # seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    if not experiment.config['load_checkpoint']:
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    group_matrix_generate_with_overlap(experiment.config)