import numpy as np
from matplotlib import cm, colors, rc
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import os
from mpl_toolkits.mplot3d import Axes3D

def vis_2dpose(location):
    rc('image', cmap='rainbow_r')
    # location = np.load("init_pose.npy")

    q = location[:,3:]
    q = q[:, [1, 2, 3, 0]]
    rpy = Rot.from_quat(q).as_euler("XYZ")
    location = np.concatenate((location[:,:3],rpy),axis=1)
    t = np.arange(location.shape[0]) / location.shape[0]
    # location[:, 0] = location[:, 0] - np.mean(location[:, 0])
    # location[:, 1] = location[:, 1] - np.mean(location[:, 1])
    u = np.cos(location[:, -1]) * 2
    v = np.sin(location[:, -1]) * 2
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    ax.quiver(location[:, 0], location[:, 1], u, v, t, scale=20, scale_units='inches', width=1e-3)
    ax.axis('equal')
    ax.tick_params(axis='both', labelsize=18)
    norm = colors.Normalize(0, location.shape[0])
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow_r'), ax=ax)
    cbar.ax.tick_params(labelsize=18)

    plt.savefig('Lnetpose.png', dpi=600)
    plt.show()


def vis_pcd(obs_initial):
    obs_initial = obs_initial.reshape(-1, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = obs_initial[:, 0]
    y = obs_initial[:, 1]
    z = obs_initial[:, 2]
    ax.scatter(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('./pcd_save/pcd.png', dpi=600)
