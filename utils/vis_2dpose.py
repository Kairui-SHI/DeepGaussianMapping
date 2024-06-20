import numpy as np
from matplotlib import cm, colors, rc
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import os
from mpl_toolkits.mplot3d import Axes3D

def plot_global_pose(location, output_dir, Init_location=None, gt_location=None, epoch=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    rc('image', cmap='winter')
    gt_color = "red"
    init_color = "black"
    gt_cmap = colors.ListedColormap([gt_color])
    init_cmap = colors.ListedColormap([init_color])

    q = location[:,3:]
    q = q[:, [1, 2, 3, 0]]
    rpy = Rot.from_quat(q).as_euler("XYZ")
    location = np.concatenate((location[:,:3],rpy),axis=1)
    t = np.arange(location.shape[0]) / location.shape[0]
    if Init_location is not None:
        init_q = Init_location[:,3:]
        init_q = init_q[:, [1, 2, 3, 0]]
        init_rpy = Rot.from_quat(init_q).as_euler("XYZ")
        Init_location = np.concatenate((Init_location[:,:3],init_rpy),axis=1)
    if gt_location is not None:
        gt_q = gt_location[:,3:]
        gt_q = gt_q[:, [1, 2, 3, 0]]
        gt_rpy = Rot.from_quat(gt_q).as_euler("XYZ")
        gt_location = np.concatenate((gt_location[:,:3],gt_rpy),axis=1)

    # xy plane
    # location[:, 0] = location[:, 0] - np.mean(location[:, 0])
    # location[:, 1] = location[:, 1] - np.mean(location[:, 1])
    u = np.cos(location[:, -1]) * 2
    v = np.sin(location[:, -1]) * 2
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=20, scale_units='inches', width=1e-3, cmap='winter')
    if Init_location is not None:
        init_u = np.cos(Init_location[:, -1]) * 2
        init_v = np.sin(Init_location[:, -1]) * 2
        ax.quiver(Init_location[:, 0], Init_location[:, 1], init_u, init_v, t, scale=20, scale_units='inches', width=1e-3, cmap=init_cmap)
    if gt_location is not None:
        gt_u = np.cos(gt_location[:, -1]) * 2
        gt_v = np.sin(gt_location[:, -1]) * 2
        ax.quiver(gt_location[:, 0], gt_location[:, 1], gt_u, gt_v, t, scale=20, scale_units='inches', width=1e-3, cmap=gt_cmap)
    ax.axis('equal')
    ax.tick_params(axis='both', labelsize=18)
    norm = colors.Normalize(0, location.shape[0])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='winter'), ax=ax)
    cbar.ax.tick_params(labelsize=18)
    ax.set_title("xy_plane_estpose", fontsize=32)
    plt.savefig(os.path.join(output_dir, "xy_plane_estpose.png"), dpi=600)
    plt.close()

    # xz plane
    u = np.cos(location[:, 4]) * 2
    v = np.sin(location[:, 4]) * 2
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    ax.quiver(location[:, 0], location[:, 2], u, v, t, scale=20, scale_units='inches', width=1e-3, cmap='winter')
    if Init_location is not None:
        init_u = np.cos(Init_location[:, 4]) * 2
        init_v = np.sin(Init_location[:, 4]) * 2
        ax.quiver(Init_location[:, 0], Init_location[:, 2], init_u, init_v, t, scale=20, scale_units='inches', width=1e-3, cmap=init_cmap)
    if gt_location is not None:
        gt_u = np.cos(gt_location[:, 4]) * 2
        gt_v = np.sin(gt_location[:, 4]) * 2
        ax.quiver(gt_location[:, 0], gt_location[:, 2], gt_u, gt_v, t, scale=20, scale_units='inches', width=1e-3, cmap=gt_cmap)
    ax.axis('equal')
    ax.tick_params(axis='both', labelsize=18)
    norm = colors.Normalize(0, location.shape[0])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='winter'), ax=ax)
    cbar.ax.tick_params(labelsize=18)
    ax.set_title("xz_plane_estpose", fontsize=32)
    plt.savefig(os.path.join(output_dir, "xz_plane_estpose.png"), dpi=600)
    plt.close()

# location = np.load("/mnt/massive/skr/SplaTAM/pcd_save/room0_0/GSsetup_Lnet_est_pose_c2w.npy")
# plot_global_pose(location, output_dir="/mnt/massive/skr/SplaTAM/results/")