import numpy as np
from tqdm import tqdm
import itertools
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from utils.slam_external import build_rotation

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