import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

from common.skeleton import Skeleton


def resample_pose_seq(poses, duration_in_sec, fps):
    n = len(poses)
    x = np.arange(0, n)
    y = poses
    f = interp1d(x, y, axis=0, kind='linear', fill_value='extrapolate')
    expected_n = duration_in_sec * fps
    x_new = np.arange(0, n, n / expected_n)
    interpolated_y = f(x_new)
    if hasattr(poses, 'dtype'):
        interpolated_y = interpolated_y.astype(poses.dtype)
    return interpolated_y


def convert_dir_vec_to_pose(vec, skeleton: Skeleton):
    vec = np.array(vec)
    assert len(vec.shape) >= 2 and vec.shape[-1] == 3
    num_joints = skeleton.num_joints()
    assert vec.shape[-2] == num_joints - 1
    ori_shape = vec.shape
    vec = vec.reshape(-1, num_joints - 1, 3)
    
    pos_shape = list(vec.shape)
    pos_shape[1] += 1
    joint_pos= np.zeros(pos_shape, dtype=vec.dtype)
    bone_lengths = skeleton.bone_lengths()
    for j, parent_j in enumerate(skeleton.parents()):
        if parent_j == -1:
            continue
        joint_pos[:, j] = joint_pos[:, parent_j] + bone_lengths[j] * vec[:, j - 1]

    target_shape = list(ori_shape)
    target_shape[-2] = num_joints
    joint_pos = joint_pos.reshape(target_shape)

    return joint_pos


def convert_pose_seq_to_dir_vec(pose, skeleton: Skeleton):
    pose = np.array(pose)
    assert len(pose.shape) >= 2 and pose.shape[-1] == 3
    num_joints = skeleton.num_joints()
    assert pose.shape[-2] == num_joints
    ori_shape = pose.shape
    pose = pose.reshape(-1, num_joints, 3)

    dir_vec = np.zeros_like(pose)
    for j, parent_j in enumerate(skeleton.parents()):
        if parent_j == -1:
            continue
        dir_vec[:, j] = pose[:, j] - pose[:, parent_j]
        dir_vec[:, j] = normalize(dir_vec[:, j], axis=1)
    target_shape = list(ori_shape)
    target_shape[-2] = num_joints - 1
    dir_vec = dir_vec[:, 1:].reshape(target_shape)

    return dir_vec
