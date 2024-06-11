import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys
_BASE_DIR = '/mnt/massive/skr/SplaTAM'
sys.path.append(_BASE_DIR)

from utils.slam_external import build_rotation
from utils.slam_helpers import matrix_to_quaternion
from datasets.gradslam_datasets import (load_dataset_config, ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
                                        ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset, TUMDataset,
                                        ScannetPPDataset, NeRFCaptureDataset)
def params2pose():
    params = np.load('/mnt/massive/skr/SplaTAM/experiments/Replica/room0_0/params.npz')
    print(params.files)
    params = {key: torch.from_numpy(value) for key, value in params.items()}

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

    init_pose = []
    gt_pose = []
    pose0 = []
    pose1 = []
    w2c = []
    init_w2c = torch.eye(4).cuda().float()

    for idx in tqdm(range(params['cam_unnorm_rots'].shape[-1])):
        gt_w2c = params['gt_w2c_all_frames'][idx]
        gt_c2w = torch.inverse(gt_w2c)

        gt_rot_quat = matrix_to_quaternion(gt_c2w[:3,:3]).reshape(1,4)
        gt_tran = gt_c2w[:3, 3].detach().reshape(1,3)
        pose0 = torch.cat((gt_tran, gt_rot_quat), dim=1).detach().cpu().numpy()

        gt_pose.append(pose0)

        '''init_pose'''
        init_w2c_quat = F.normalize(params['cam_unnorm_rots'][..., idx])
        init_w2c_rot = build_rotation(init_w2c_quat)
        init_w2c_tran = params['cam_trans'][..., idx]
        init_w2c[:3,:3] =  init_w2c_rot
        init_w2c[:3,3] = init_w2c_tran
        init_c2w = torch.inverse(init_w2c)

        init_rot_quat = matrix_to_quaternion(init_c2w[:3,:3]).reshape(1,4)
        init_tran = init_c2w[:3, 3].detach().reshape(1,3)
        pose1 = torch.cat((init_tran, init_rot_quat), dim=1).detach().cpu().numpy()

        init_pose.append(pose1)

    gt_pose= np.concatenate(gt_pose, axis=0)
    np.save("./pcd_save/replica/pcd/gt_pose.npy", gt_pose)
    print(gt_pose.shape)

    init_pose= np.concatenate(init_pose, axis=0)
    np.save("./pcd_save/replica/prior/init_pose.npy", init_pose)
    print(init_pose.shape)

def Lnet2pose(location):
    w2c = torch.eye(4).cuda().float()
    location = torch.from_numpy(location)
    location.shape

    Lnet_pose_est = []
    for idx in tqdm(range(location.shape[0])):
        w2c_rot = build_rotation(F.normalize(location[idx, 3:].reshape(1,4)))
        w2c_tran = location[idx, :3]
        w2c[:3,:3] =  w2c_rot
        w2c[:3,3] = w2c_tran
        c2w = torch.inverse(w2c)
        # c2w = w2c
        quat_rot = matrix_to_quaternion(c2w[:3,:3]).reshape(1,4)
        quat_tran = c2w[:3, 3].detach().reshape(1,3)
        pose = torch.cat((quat_tran, quat_rot), dim=1).detach().cpu().numpy()
        Lnet_pose_est.append(pose)

    Lnet_pose_est= np.concatenate(Lnet_pose_est, axis=0)
    np.save("./pcd_save/c2w_Lnet_pose_est.npy", Lnet_pose_est)
    print(Lnet_pose_est.shape)

def w2c_to_c2w(location):
    w2c = torch.eye(4).cuda().float()
    c2w_pose = torch.empty(location.shape[0], 7)
    for idx in range(location.shape[0]):
        w2c_rot = build_rotation(F.normalize(location[idx, 3:].reshape(1,4)))
        w2c_tran = location[idx, :3]
        w2c[:3,:3] =  w2c_rot
        w2c[:3,3] = w2c_tran
        c2w = torch.inverse(w2c)
        # c2w = w2c
        quat_rot = matrix_to_quaternion(c2w[:3,:3]).reshape(1,4)
        quat_tran = c2w[:3, 3].detach().reshape(1,3)
        pose = torch.cat((quat_tran, quat_rot), dim=1)
        c2w_pose[idx] = pose

    return c2w_pose

def reference_trans(location):
    first_c2w_trans = location[0, :3]
    first_c2w_quat = location[0, 3:]
    first_c2w_rot = build_rotation(F.normalize(first_c2w_quat.reshape(1,4)))
    transformation = torch.eye(4).cuda().float()
    transformation[:3,:3] = first_c2w_rot
    transformation[:3,3] = first_c2w_trans
    transformation = torch.inverse(transformation)

    reference_trans_pose = torch.empty(location.shape[0], 7)
    c2w = torch.eye(4).cuda().float()
    for idx in range(location.shape[0]):
        c2w_trans = location[idx, :3]
        c2w_quat = location[idx, 3:]
        c2w_rot = build_rotation(F.normalize(c2w_quat.reshape(1,4)))
        c2w[:3,:3] =  c2w_rot
        c2w[:3,3] = c2w_trans

        reference_trans_matrix = transformation @ c2w
        quat_rot = matrix_to_quaternion(reference_trans_matrix[:3,:3]).reshape(1,4)
        quat_tran = reference_trans_matrix[:3, 3].detach().reshape(1,3)
        pose = torch.cat((quat_tran, quat_rot), dim=1)
        reference_trans_pose[idx] = pose

    return reference_trans_pose

# # # location = np.load("/mnt/massive/skr/SplaTAM/pcd_save/apartment_0/Lnet_est_pose_00.npy")
# location = np.load("./pcd_save/room0_0/GSsetup_Lnet_est_pose_eval.npy")
# # location = np.loadtxt("/mnt/massive/skr/SplaTAM/pcd_save/room0_0/gt_all_frames.txt")
# # # Initial_pose = np.load("/mnt/massive/skr/SplaTAM/pcd_save/room0_0/Initial_pose.npy")
# location = torch.from_numpy(location)
# # # # Initial_pose = torch.from_numpy(Initial_pose)
# # location = w2c_to_c2w(location)
# # # # Initial_pose = w2c_to_c2w(Initial_pose)
# location = reference_trans(location)
# np.save("/mnt/massive/skr/SplaTAM/pcd_save/room0_0/GSsetup_Lnet_est_pose_eval_reference.npy", location)

# # params = np.load('/mnt/massive/skr/SplaTAM/experiments/Apartment/Post_SplaTAM_Opt/params.npz', allow_pickle=True)
# # print(params)
# # print(params.files)


# gt_pose = np.load(f"./pcd_save/room0_0/gt_all_frames_c2w.npy")
# print(gt_pose)
