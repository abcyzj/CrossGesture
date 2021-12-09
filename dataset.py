import argparse
import os
import pickle

import lmdb
import numpy as np
import torch
from sklearn.preprocessing import normalize

from common.data_utils import (convert_dir_vec_to_pose,
                               convert_pose_seq_to_dir_vec)
from common.mocap_dataset import MocapDataset
from common.skeleton import Skeleton
from vis import render_animation

BAIJIA_SKELETON = Skeleton(
    parents=[-1,  0,  1,  2,  3,  1,  5,  6,  1,  8,  9],
    joints_left=[5, 6, 7],
    joints_right=[8, 9, 10],
    bone_lengths=None
)

BAIJIA_CAMERA = {
    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70, # Only used for visualization
    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}

class BaijiaDataset(MocapDataset):
    def __init__(self, args, is_train):
        super().__init__(25, BAIJIA_SKELETON, BAIJIA_CAMERA)

        self.base_path = args.data_path
        self.seq_len = args.seq_len
        self.seq_stride = args.seq_stride
        self.is_train = is_train
        self.db_env = None
        self.db_txn = None

    def __del__(self):
        if self.db_txn is not None:
            self.db_txn.abort()

    def __len__(self):
        if self.db_env is None:
            self._init_lmdb()
        if self.is_train:
            return len(self.expanded_train_keys)
        else:
            return len(self.expanded_test_keys)

    def __getitem__(self, index):
        if self.db_env is None:
            self._init_lmdb()
        if self.is_train:
            cur_key = self.expanded_train_keys[index]
        else:
            cur_key = self.expanded_test_keys[index]
        keypoints_3d = self.pose_seq_list[self.key2pose_index[cur_key]]
        cur_len = keypoints_3d.shape[0]
        if self.is_train:
            start_frame = int((cur_len - self.seq_len) * torch.rand([]).item())
        else: # For test, clip sampling must not be random
            in_key_index = index - self.key2expanded_start_index[cur_key]
            start_frame = in_key_index * self.seq_stride
        end_frame = start_frame + self.seq_len
        return self.keypoints_to_normalized_dir_vec(keypoints_3d[start_frame:end_frame])

    def _init_lmdb(self):
        self.db_env = lmdb.open(os.path.join(self.base_path, 'baijia_upper'), map_size=20*1024**3, readonly=True, lock=False, max_dbs=64)
        self.db_txn = self.db_env.begin(write=False)
        db_keys = list(self.db_txn.cursor().iternext(values=False))
        rng = np.random.RandomState(8848)
        self.test_keys = set(rng.choice(db_keys, int(len(db_keys) * 0.1), replace=False)) # By default, take 10% for testing
        self.train_keys = set(db_keys) - set(self.test_keys)
        self.all_keys = db_keys

        self.expanded_keys = []
        self.expanded_test_keys = []
        self.expanded_train_keys = []
        self.key2pose_index = {}
        self.key2expanded_start_index = {}
        self.pose_seq_list = []
        for key in self.all_keys:
            val = pickle.loads(self.db_txn.get(key))
            cur_len = val['keypoints_3d'].shape[0]
            expand_num = (cur_len - self.seq_len) // self.seq_stride + 1 if cur_len >= self.seq_len else 0
            self.expanded_keys.extend([key] * expand_num)
            self.key2pose_index[key] = len(self.pose_seq_list)
            self.key2expanded_start_index[key] = len(self.expanded_test_keys)
            if key in self.train_keys:
                self.expanded_train_keys.extend([key] * expand_num)
            else:
                self.expanded_test_keys.extend([key] * expand_num)
            self.pose_seq_list.append(val['keypoints_3d'])
        self.pose_seq_list = self.pose_seq_list
        all_pose_frames = np.vstack(self.pose_seq_list)
        bone_lengths = []
        for j, parent_j in enumerate(self._skeleton.parents()):
            if parent_j == -1:
                bone_lengths.append(0)
            else:
                bones = all_pose_frames[:, j] - all_pose_frames[:, parent_j]
                mean_bone_length = np.mean(np.linalg.norm(bones, axis=1))
                bone_lengths.append(mean_bone_length)
        self._skeleton._bone_lengths = np.array(bone_lengths)
        dir_vec = convert_pose_seq_to_dir_vec(all_pose_frames, self._skeleton)
        mean_dir_vec = np.mean(dir_vec, axis=0)
        mean_dir_vec = normalize(mean_dir_vec, axis=1)[np.newaxis, :]
        self.mean_dir_vec = mean_dir_vec
        print(f'Baijia LMDB initialized, {len(self.all_keys)} keys, {len(self.expanded_keys)} samples, {len(self.train_keys)} keys for train, {len(self.test_keys)} keys for test.')

    def keypoints_to_normalized_dir_vec(self, keypoints):
        dir_vec = convert_pose_seq_to_dir_vec(keypoints, self._skeleton)
        return dir_vec - self.mean_dir_vec

    def normalized_dir_vec_to_keypoints(self, dir_vec):
        dir_vec = dir_vec + self.mean_dir_vec
        return convert_dir_vec_to_pose(dir_vec, self._skeleton)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', type=int, default=100)
    parser.add_argument('--clip_stride', default=10)
    parser.add_argument('--base_path', default='data/')
    parser.add_argument('--is_train', type=bool, default=False)
    args = parser.parse_args()
    dataset = BaijiaDataset(args)
    pose = dataset[10]
    dir_vec = dataset.keypoints_to_normalized_dir_vec(pose)
    recon_pose = dataset.normalized_dir_vec_to_keypoints(dir_vec)
    pose = dataset.camera_to_world(pose)
    recon_pose = dataset.camera_to_world(recon_pose)
    poses = {'Main': pose, 'Recon': recon_pose}
    render_animation(poses, dataset.skeleton(), dataset.fps(), 3000, dataset.camera()['azimuth'], 'test4.mp4')
