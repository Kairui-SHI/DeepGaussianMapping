# Copyright (C) 2022-2023, NYU AI4CE Lab. All rights reserved.


import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import LocNetRegKITTI, MLP
from utils.geometry_utils import transform_to_global_KITTI, compose_pose_diff, euler_pose_to_quaternion,quaternion_to_euler_pose, qmul_torch
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from init_pose import w2c_to_c2w

def rendering_loss(params, curr_data, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, depth_weighted_bias = 0.5, tracking = True):
    # Initialize Loss Dictionary
    losses = {}

    # Initialize Render Variables
    transformed_gaussians = transform_to_frame(params, iter_time_idx, gaussians_grad=False, camera_grad=True)
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'], transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    
    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Weighted Depth for far-near depth sensor error 
    bias = depth_weighted_bias
    depth_weighted = 1.0 / (curr_data['depth'] + bias)

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = (torch.abs(curr_data['depth'] - depth)*depth_weighted)[mask].sum()
        else:
            losses['depth'] = (torch.abs(curr_data['depth'] - depth)*depth_weighted)[mask].mean()
    
    # RGB Loss
    # losses['im'] = torch.abs(curr_data['im'] - im).sum()
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    # seen = radius > 0
    # variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    # variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss


class DeepMapping2(nn.Module):
    #def __init__(self, loss_fn, n_samples=35, dim=[3, 256, 256, 256, 256, 256, 256, 1]):
    def __init__(self, n_points, rotation_representation='quaternion', device='cuda', dim=[3, 64, 256, 256, 64, 1]):
        super(DeepMapping2, self).__init__()
        # self.n_samples = n_samples
        # self.loss_fn = loss_fn
        self.n_points = n_points
        self.device = device
        self.rotation = rotation_representation
        if self.rotation == 'quaternion':
            self.loc_net = LocNetRegKITTI(n_points=n_points, out_dims=7) # <x,y,z,theta>
        else:
            self.loc_net = LocNetRegKITTI(n_points=n_points, out_dims=6) # <x,y,z,theta>
        # self.occup_net = MLP(dim)
        # self.alpha = alpha
        # self.beta = beta
       

    def forward(self, obs_local, params, group_matrix, group_data=None, variables=None, loss_init=10, flag=True,  params_init=None):
        # obs_local: <Group x Number x 3> 

        # get sensor_pose : G * 7
        with torch.no_grad():
            sensor_pose = torch.empty(len(group_matrix), 7)
            for i in range(len(group_matrix)):
                sensor_pose[i] = torch.cat((params['cam_trans'][..., group_matrix[i]], params['cam_unnorm_rots'][..., group_matrix[i]]), dim=-1).unsqueeze(0)
                # sensor_pose[i] = params['sensor_pose'][group_matrix[i]].unsqueeze(0)
            sensor_pose = sensor_pose.to(self.device)
            self.sensor_pose_c2w = w2c_to_c2w(sensor_pose).to(self.device)
            # print(sensor_pose)

        # G = obs_local.shape[0]
        self.obs_local = obs_local
        # if self.rotation == 'quaternion':
        #     sensor_pose = euler_pose_to_quaternion(sensor_pose)
        self.obs_initial = transform_to_global_KITTI(
            self.sensor_pose_c2w , self.obs_local, rotation_representation=self.rotation)
        # vis_pcd(self.obs_initial.detach().cpu().numpy()) # visualize one of frame's pcd
        self.l_net_out = self.loc_net(self.obs_initial)
        if self.rotation == 'quaternion':
            original_shape = list(sensor_pose.shape)
            xyz = self.l_net_out[:,:3]+ sensor_pose[:,:3]
            wxyz = qmul_torch(self.l_net_out[:,3:], sensor_pose[:,3:])
            self.pose_est =  torch.cat((xyz, wxyz), dim=1).view(original_shape)
        elif self.rotation == 'euler_angle':
            self.pose_est = self.l_net_out + sensor_pose

        self.obs_global_est = transform_to_global_KITTI(
            self.pose_est, self.obs_local, rotation_representation=self.rotation)
    
        # pose_est= G*7 -> params
        self.params = {k: v.clone() for k, v in params.items()}
        for i in range(len(group_matrix)):
            self.params['cam_unnorm_rots'][..., group_matrix[i]] =self.pose_est[i, 3:].float() #quaternion_rot 1*4
            self.params['cam_trans'][..., group_matrix[i]] = self.pose_est[i, :3].float() #quaternion_tran 1*3

        # if not self.training:
        #     self.pose_est = self.pose_est
            # rel_w2c = group_data[0]['iter_gt_w2c_list'][-1]
            # rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0)
            # rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            # rel_w2c_tran = rel_w2c[:3, 3].detach()
            # params_gt = {k: v.clone() for k, v in params.items()}
            # params_gt['cam_unnorm_rots'][..., group_matrix[0]] = rel_w2c_rot_quat.float() #quaternion_rot 1*4
            # params_gt['cam_trans'][..., group_matrix[0]] = rel_w2c_tran.float() #quaternion_tran 1*3

            # loss_est = rendering_loss(self.params_est, group_data[0], group_matrix[0], loss_weights=dict(im=0.5,depth=1.0,), use_sil_for_loss=True, sil_thres=0.99, use_l1=True, ignore_outlier_depth_loss=False)
            # loss = rendering_loss(params, group_data[0], group_matrix[0], loss_weights=dict(im=0.5,depth=1.0,), use_sil_for_loss=True, sil_thres=0.99, use_l1=True, ignore_outlier_depth_loss=False)
            # loss_gt = rendering_loss(params_gt, group_data[0], group_matrix[0], loss_weights=dict(im=0.5,depth=1.0,), use_sil_for_loss=True, sil_thres=0.99, use_l1=True, ignore_outlier_depth_loss=False)
            # print(loss_est)
            # print(loss)
            # print(loss_gt)
            # return loss_est, loss

        if self.training: # Get Loss: Gaussians  
            loss = 0
            loss = rendering_loss(self.params, group_data[i], group_matrix[i], loss_weights=dict(im=0.5,depth=1.0,), use_sil_for_loss=True, sil_thres=0.99, use_l1=True, ignore_outlier_depth_loss=False, depth_weighted_bias=0.5)
            loss = loss*0.1
            print(loss)

            if params_init is not None:
                loss_init = rendering_loss(params, group_data[0], group_matrix[0], loss_weights=dict(im=0.5,depth=1.0,), use_sil_for_loss=True, sil_thres=0.99, use_l1=True, ignore_outlier_depth_loss=False, depth_weighted_bias=0.5)
                loss_init = loss_init*1e-1
                print(loss_init)
            # rel_w2c = group_data[0]['iter_gt_w2c_list'][-1]
            # rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0)
            # rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            # rel_w2c_tran = rel_w2c[:3, 3].detach()
            # params_gt = {k: v.clone() for k, v in params.items()}
            # params_gt['cam_unnorm_rots'][..., group_matrix[0]] = rel_w2c_rot_quat.float() #quaternion_rot 1*4
            # params_gt['cam_trans'][..., group_matrix[0]] = rel_w2c_tran.float() #quaternion_tran 1*3
            # loss_gt = rendering_loss(params_gt, group_data[0], group_matrix[0], loss_weights=dict(im=0.5,depth=1.0,), use_sil_for_loss=True, sil_thres=0.99, use_l1=True, ignore_outlier_depth_loss=False, depth_weighted_bias=0.5)
            # print(loss_gt)

            threshold = 0.2
            loss_pose_ = []
            for i in range(len(group_matrix)):
                distanceT = torch.abs(sensor_pose[i, :3] - self.pose_est[i, :3]).sum()
                distanceR = torch.abs(F.normalize(sensor_pose[i, 3:].unsqueeze(0)) - F.normalize(self.pose_est[i, 3:].unsqueeze(0))).sum()
                distance = distanceT + distanceR
                print(distanceT, distanceR)
                # distance = distance_cal(self.pose_est[i], sensor_pose[i])
                loss_pose_.append(distance)
            loss_pose = sum(loss_pose_) / len(loss_pose_)
            print(loss_pose)

            if loss_pose < threshold:
                # print('\033[91m' + 'init_pose track!' + '\033[0m')
                loss = loss #+ distanceR*1e6
                flag = 1
            elif loss_pose > threshold:
                loss = loss + loss_pose*1e5
                # loss = loss_pose*1e5
                flag = 0

            # # latter-train
            # flag = 1
            # loss = loss #+ loss_pose*1e5

            return loss_init, flag, loss

    # def compute_loss(self):
    #     valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
    #     bce_weight = torch.cat(
    #         (self.valid_points, valid_unoccupied_points), 1).float()
    #     # <Bx(n+1)Lx1> same as occp_prob and gt
    #     bce_weight = bce_weight.unsqueeze(-1)

    #     if self.loss_fn.__name__ == 'bce_ch':
    #         loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
    #                             self.valid_points, bce_weight, seq=2, gamma=1-self.alpha)  # BCE_CH
    #     elif self.loss_fn.__name__ == 'bce':
    #         loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
    #     elif self.loss_fn.__name__ == 'bce_ch_eu':
    #         loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est, self.relative_centroid, self.centorid,
    #                             self.valid_points, bce_weight, seq=2, alpha=self.alpha, beta=self.beta)
    #     elif self.loss_fn.__name__ == 'pose':
    #         loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est, self.t_src, self.t_dst, self.r_src, self.r_dst,
    #                             self.valid_points, bce_weight, seq=2, alpha=self.alpha, beta=self.beta)
    #     return loss
