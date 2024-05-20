import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from slam_external import build_rotation
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# from utils.geometry_utils import transform_to_global_KITTI

def compute_entropy(covariance):
    entropy = 0.5 * np.log(2 * np.pi * np.e * np.linalg.det(covariance))
    return entropy

def compute_mean_map_entropy(pointcloud, radius):
    kdtree = o3d.geometry.KDTreeFlann(pointcloud)

    mean_entropy = 0.0
    valid_points = 0
    entropies = np.zeros((len(pointcloud.points),))
    valid_entropy_points = np.zeros((len(pointcloud.points),), dtype=bool)

    for i in tqdm(range(len(pointcloud.points))):
        [k, idx, _] = kdtree.search_radius_vector_3d(pointcloud.points[i], radius)
        
        if k > 10:
            points = np.asarray(pointcloud.points)[idx, :]
            mean = points.mean(axis=0)
            centered = points - mean
            covariance = np.cov(centered, rowvar=False)
            entropy = compute_entropy(covariance)

            if np.isfinite(entropy):
                mean_entropy += entropy
                entropies[i] = entropy
                valid_entropy_points[i] = True
                valid_points += 1
                # print(f"Entropy {entropy}")

    if valid_points > 0:
        mean_entropy /= valid_points

    print(f"MME Valid_points {valid_points * 100.0 / len(pointcloud.points)}% {valid_points} {len(pointcloud.points)}")
    
    return mean_entropy, entropies

def color_point_cloud_by_mme(point_cloud, entropies, min_abs_entropy, max_abs_entropy):
    colored_point_cloud = o3d.geometry.PointCloud()
    for i in tqdm(range(len(entropies))):
        if np.abs(entropies[i]) > max_abs_entropy: 
            normalized_entropy = max_abs_entropy
        elif np.abs(entropies[i]) < min_abs_entropy:
            normalized_entropy = min_abs_entropy
        else:
            normalized_entropy = (np.abs(entropies[i]) - min_abs_entropy) / (max_abs_entropy - min_abs_entropy)
        color = cm.jet(normalized_entropy)[:3]
        colored_point_cloud.points.append(point_cloud.points[i])
        colored_point_cloud.colors.append(color)
    return colored_point_cloud

print('load pcd ...')
gt_pcd = o3d.io.read_point_cloud('./pcd_save/gt_pcd.ply')
Lnet_est_pcd = o3d.io.read_point_cloud('./pcd_save/Lnet_pcd_est.ply')
init_pcd = o3d.io.read_point_cloud('./pcd_save/init_pcd.ply')
# o3d.visualization.draw_geometries([pointcloud])

# print('compute mean map entropy ...')
# radius = 0.1
# # est_mme, est_entropies = compute_mean_map_entropy(Lnet_est_pcd, radius)
# # gt_mme, gt_entropies = compute_mean_map_entropy(gt_pcd, radius)
# # init_mme, init_entropies = compute_mean_map_entropy(init_pcd, radius)

# data = np.load('pcd_save/mean_map_entropy.npz')
# est_entropies = data['est_entropies']
# gt_entropies = data['gt_entropies']
# est_mme = data['est_mme']
# gt_mme = data['gt_mme']
# np.savez('pcd_save/mean_map_entropy.npz', est_entropies=est_entropies, gt_entropies=gt_entropies, init_entropies=init_entropies, est_mme=est_mme, gt_mme=gt_mme, init_mme=init_mme)

# print(f"Estimated MME: {est_mme}")
# print(f"GT MME: {gt_mme}")
# print(f"Init MME: {init_mme}")

print('color point cloud by mme ...')
data = np.load('pcd_save/mean_map_entropy.npz')
est_entropies = data['est_entropies']
gt_entropies = data['gt_entropies']
init_entropies = data['init_entropies']
est_mme = data['est_mme']
gt_mme = data['gt_mme']
init_mme = data['init_mme']
# # Remove zeros from the arrays
# est_entropies = est_entropies[est_entropies != 0]
# gt_entropies = gt_entropies[gt_entropies != 0]
# init_entropies = init_entropies[init_entropies != 0]

print(f"Estimated MME: {est_mme}")
print(f"GT MME: {gt_mme}")
print(f"Init MME: {init_mme}")
entropies = np.concatenate((est_entropies, gt_entropies, init_entropies))
# min_abs_entropy = np.min(np.abs(entropies))
# max_abs_entropy = np.max(np.abs(entropies))
min_abs_entropy = 7.5
max_abs_entropy = 13.0

print(f"Min Abs Entropy: {min_abs_entropy}")
print(f"Max Abs Entropy: {max_abs_entropy}")
# Create a colorbar
fig, ax = plt.subplots()
cmap = plt.cm.jet
norm = plt.Normalize(vmin=min_abs_entropy, vmax=max_abs_entropy)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Set an empty array to avoid error
fig.colorbar(sm, ax=ax)
plt.show()
# init
colored_point_cloud = o3d.geometry.PointCloud()
colored_point_cloud = color_point_cloud_by_mme(init_pcd, init_entropies, min_abs_entropy, max_abs_entropy)
o3d.visualization.draw_geometries([colored_point_cloud])
o3d.io.write_point_cloud("./pcd_save/mme_init_pcd.ply", colored_point_cloud)
# est
colored_point_cloud = o3d.geometry.PointCloud()
colored_point_cloud = color_point_cloud_by_mme(Lnet_est_pcd, est_entropies, min_abs_entropy, max_abs_entropy)
o3d.visualization.draw_geometries([colored_point_cloud])
o3d.io.write_point_cloud("./pcd_save/mme_est_pcd.ply", colored_point_cloud)
# gt
colored_point_cloud = o3d.geometry.PointCloud()
colored_point_cloud = color_point_cloud_by_mme(gt_pcd, gt_entropies, min_abs_entropy, max_abs_entropy)
o3d.visualization.draw_geometries([colored_point_cloud])
o3d.io.write_point_cloud("./pcd_save/mme_gt_pcd.ply", colored_point_cloud)