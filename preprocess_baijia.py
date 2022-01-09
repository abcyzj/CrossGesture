import pickle

import librosa
import lmdb
import torch
import torchaudio as ta

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
tgt_db_dnv = lmdb.open('data/baijia_audio', map_size=20*1024**3)


with src_db_env.begin() as src_txn:
    with tgt_db_dnv.begin(write=True) as tgt_txn:
        for key, src_val in src_txn.cursor():
            src_val = pickle.loads(src_val)
            keypoints_3d = src_val['keypoints_3d']
            upper_keypoints = keypoints_3d[:, valid_joint_ids]
            tgt_val = src_val
            tgt_val.pop('keypoints')
            tgt_val['keypoints_3d'] = upper_keypoints
            sr = src_val['audio_sr']
            waveform = src_val['audio_wav']
            audio_start_frame, audio_end_frame = src_val['audio_timestamp']
            keypoint_start_frame, keypoint_end_frame = src_val['keypoint_timestamp']
            sample_per_frame = sr // 25
            waveform = waveform[(keypoint_start_frame-audio_start_frame)*sample_per_frame:(keypoint_end_frame-audio_start_frame)*sample_per_frame].copy()
            tgt_val['audio_wav'] = waveform
            tgt_val.pop('audio_timestamp')
            norm_wave = torch.from_numpy(waveform.copy()).transpose(0, 1).to(torch.float32)
            if sr != 16000:
                norm_wave = ta.transforms.Resample(sr, 16000)(norm_wave)
            if norm_wave.shape[0] > 1:
                norm_wave = torch.mean(norm_wave, dim=0)
            spec = librosa.feature.melspectrogram(y=norm_wave.numpy(), sr=16000, n_fft=2048, win_length=800, hop_length=160, n_mels=80)
            spec_db = librosa.power_to_db(spec)
            tgt_val['spec'] = spec
            tgt_txn.put(key, pickle.dumps(tgt_val))
            print(key.decode())
