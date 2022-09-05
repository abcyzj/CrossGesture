import itertools
import pickle

import librosa
import lmdb
import numpy as np
import torch
import torchaudio as ta

from common.skeleton import Skeleton
from bert import ChineeseBert

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


def get_word_embedding(val, bert_model):
    FPS = 25
    scripts = val['scripts']
    lines = []
    for line in scripts:
        lines.append(''.join([interval.label for interval in line]))
    embedding = bert_model.get_bert_embedding(lines)
    flattened_embedding = []
    for i, line in enumerate(lines):
        flattened_embedding.append(embedding[i, 1:len(line)+1])
    flattened_embedding = np.concatenate(flattened_embedding, axis=0)
    flattened_interval = list(itertools.chain(*scripts))
    keypoint_start_frame, keypoint_end_frame = val['keypoint_timestamp']
    regulated_embedding = np.zeros([keypoint_end_frame - keypoint_start_frame, flattened_embedding.shape[1]], dtype=flattened_embedding.dtype)
    silence_frame_ind = np.zeros([keypoint_end_frame - keypoint_start_frame, 1], dtype=np.int8)
    cur_interval_ind = 0
    regulate_finished = False
    for frame_num in range(keypoint_start_frame, keypoint_end_frame):
        cur_frame_ind = frame_num - keypoint_start_frame
        cur_interval = flattened_interval[cur_interval_ind]
        while frame_num >= cur_interval.end * FPS:
            cur_interval_ind += 1
            if cur_interval_ind >= len(flattened_interval):
                regulate_finished = True
                break
            cur_interval = flattened_interval[cur_interval_ind]
        if regulate_finished:
            break

        if frame_num >= cur_interval.start * FPS and frame_num < cur_interval.end * FPS:  # 当前帧在当前interval内部
            regulated_embedding[cur_frame_ind] = flattened_embedding[cur_interval_ind]
        elif frame_num < cur_interval.start * FPS:  # 当前帧在interval左侧
            regulated_embedding[cur_frame_ind] = 0
            silence_frame_ind[cur_frame_ind] = 1
    return regulated_embedding, silence_frame_ind


if __name__ == '__main__':
    src_db_env = lmdb.open('data/baijia_audio', readonly=True, map_size=20*1024**3)
    tgt_db_dnv = lmdb.open('data/baijia_all', map_size=40*1024**3)

    bert_model = ChineeseBert(device=torch.device('cuda'))


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
                # tgt_val = src_val.copy()
                # regulated_embedding, silence_frame_ind = get_word_embedding(src_val, bert_model)
                # tgt_val['word_embedding'] = regulated_embedding
                # tgt_val['silence'] = silence_frame_ind
                # tgt_txn.put(key, pickle.dumps(tgt_val))
                # print(key.decode())
