import pickle

import lmdb

from common.skeleton import Skeleton

FULL_SKELETON = Skeleton(
    parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  7, 11, 12,  7, 14, 15],
    joints_left=[4, 5, 6, 11, 12, 13],
    joints_right=[1, 2, 3, 14, 15, 16],
    bone_lengths=None
)

valid_joint_ids = [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

UPPER_SKELETON = Skeleton(
    parents=[-1,  0,  1,  2,  3,  1,  5,  6,  1,  8,  9],
    joints_left=[5, 6, 7],
    joints_right=[8, 9, 10],
    bone_lengths=None
)

src_db_env = lmdb.open('data/baijia', readonly=True, map_size=20*1024**3)
tgt_db_dnv = lmdb.open('data/baijia_upper', map_size=20*1024**3)


with src_db_env.begin() as src_txn:
    with tgt_db_dnv.begin(write=True) as tgt_txn:
        for key, src_val in src_txn.cursor():
            src_val = pickle.loads(src_val)
            keypoints_3d = src_val['keypoints_3d']
            upper_keypoints = keypoints_3d[:, valid_joint_ids]
            tgt_val = src_val
            tgt_val.pop('keypoints')
            tgt_val['keypoints_3d'] = upper_keypoints
            tgt_txn.put(key, pickle.dumps(tgt_val))
            print(key.decode())
