import datetime
import logging
import os
import pickle
import random

import numpy as np
import lmdb as lmdb
import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import utils.data_utils
from model.vocab import Vocab
from data_loader.data_preprocessor import DataPreprocessor
import pyarrow


def word_seq_collate_fn(data):
    """ collate function for loading word sequences in variable lengths """
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    word_seq, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    # merge sequences
    words_lengths = torch.LongTensor([len(x) for x in word_seq])
    word_seq = pad_sequence(word_seq, batch_first=True).long()

    text_padded = default_collate(text_padded)
    poses_seq = default_collate(poses_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, spectrogram, aux_info


def default_collate_fn(data):
    _, text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info = zip(*data)

    text_padded = default_collate(text_padded)
    pose_seq = default_collate(pose_seq)
    vec_seq = default_collate(vec_seq)
    audio = default_collate(audio)
    spectrogram = default_collate(spectrogram)
    aux_info = {key: default_collate([d[key] for d in aux_info]) for key in aux_info[0]}

    return torch.tensor([0]), torch.tensor([0]), text_padded, pose_seq, vec_seq, audio, spectrogram, aux_info


class SpeechMotionDataset(Dataset):
    mean_dir_vec = np.array([ 0.0154009, -0.9690125, -0.0884354, -0.0022264, -0.8655276, 0.4342174, -0.0035145, -0.8755367, -0.4121039, -0.9236511, 0.3061306, -0.0012415, -0.5155854,  0.8129665,  0.0871897, 0.2348464,  0.1846561,  0.8091402,  0.9271948,  0.2960011, -0.013189 ,  0.5233978,  0.8092403,  0.0725451, -0.2037076, 0.1924306,  0.8196916])
    mean_pose = np.array([ 0.0000306,  0.0004946,  0.0008437,  0.0033759, -0.2051629, -0.0143453,  0.0031566, -0.3054764,  0.0411491,  0.0029072, -0.4254303, -0.001311 , -0.1458413, -0.1505532, -0.0138192, -0.2835603,  0.0670333,  0.0107002, -0.2280813,  0.112117 , 0.2087789,  0.1523502, -0.1521499, -0.0161503,  0.291909 , 0.0644232,  0.0040145,  0.2452035,  0.1115339,  0.2051307]).reshape(-1, 3)

    def __init__(self, args, is_train, speaker_model=None, remove_word_timing=False):
        self.lmdb_dir = '/home/yezj/Gesture-Generation-from-Trimodal-Context/data/ted_dataset/lmdb_train' if is_train else '/home/yezj/Gesture-Generation-from-Trimodal-Context/data/ted_dataset/lmdb_val'
        self.n_poses = args.seq_len if is_train else args.inf_seq_len
        if args.prior_seed_len:
            self.n_poses += args.prior_seed_len
        self.subdivision_stride = args.seq_stride
        self.skeleton_resampling_fps = 15
        self.remove_word_timing = remove_word_timing
        self.is_train = is_train

        self.expected_audio_length = int(round(self.n_poses / self.skeleton_resampling_fps * 16000))
        self.expected_spectrogram_length = utils.data_utils.calc_spectrogram_length_from_motion_length(
            self.n_poses, self.skeleton_resampling_fps)

        self.lang_model = None

        logging.info("Reading data '{}'...".format(self.lmdb_dir))
        preloaded_dir = self.lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            logging.info('Creating the dataset cache...')
            if self.mean_dir_vec.shape[-1] != 3:
                self.mean_dir_vec = self.mean_dir_vec.reshape(self.mean_dir_vec.shape[:-1] + (-1, 3))
            n_poses_extended = int(round(self.n_poses * 1.25))  # some margin
            data_sampler = DataPreprocessor(self.lmdb_dir, preloaded_dir, n_poses_extended, self.subdivision_stride, self.skeleton_resampling_fps, self.mean_pose, self.mean_dir_vec)
            data_sampler.run()
        else:
            logging.info('Found the cache {}'.format(preloaded_dir))

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

        # make a speaker model
        if speaker_model is None or speaker_model == 0:
            precomputed_model = self.lmdb_dir + '_speaker_model.pkl'
            if not os.path.exists(precomputed_model):
                self._make_speaker_model(self.lmdb_dir, precomputed_model)
            else:
                with open(precomputed_model, 'rb') as f:
                    self.speaker_model = pickle.load(f)
        else:
            self.speaker_model = speaker_model

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)
            sample = pyarrow.deserialize(sample)
            word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info = sample

        def extend_word_seq(lang, words, end_time=None):
            n_frames = self.n_poses
            if end_time is None:
                end_time = aux_info['end_time']
            frame_duration = (end_time - aux_info['start_time']) / n_frames

            extended_word_indices = np.zeros(n_frames)  # zero is the index of padding token
            if self.remove_word_timing:
                n_words = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        n_words += 1
                space = int(n_frames / (n_words + 1))
                for i in range(n_words):
                    idx = (i+1) * space
                    extended_word_indices[idx] = lang.get_word_index(words[i][0])
            else:
                prev_idx = 0
                for word in words:
                    idx = max(0, int(np.floor((word[1] - aux_info['start_time']) / frame_duration)))
                    if idx < n_frames:
                        extended_word_indices[idx] = lang.get_word_index(word[0])
                        # extended_word_indices[prev_idx:idx+1] = lang.get_word_index(word[0])
                        prev_idx = idx
            return torch.Tensor(extended_word_indices).long()

        def words_to_tensor(lang, words, end_time=None):
            indexes = [lang.SOS_token]
            for word in words:
                if end_time is not None and word[1] > end_time:
                    break
                indexes.append(lang.get_word_index(word[0]))
            indexes.append(lang.EOS_token)
            return torch.Tensor(indexes).long()

        duration = aux_info['end_time'] - aux_info['start_time']
        do_clipping = True

        if do_clipping:
            sample_end_time = aux_info['start_time'] + duration * self.n_poses / vec_seq.shape[0]
            audio = utils.data_utils.make_audio_fixed_length(audio, self.expected_audio_length)
            spectrogram = spectrogram[:, 0:self.expected_spectrogram_length]
            vec_seq = vec_seq[0:self.n_poses]
            pose_seq = pose_seq[0:self.n_poses]
        else:
            sample_end_time = None

        # to tensors
        word_seq_tensor = words_to_tensor(self.lang_model, word_seq, sample_end_time)
        extended_word_seq = extend_word_seq(self.lang_model, word_seq, sample_end_time)
        vec_seq = torch.from_numpy(vec_seq.copy()).float()
        pose_seq = torch.from_numpy(pose_seq.copy()).float()
        audio = torch.from_numpy(audio.copy()).float()
        spectrogram = torch.from_numpy(spectrogram.copy())

        return word_seq_tensor, extended_word_seq, pose_seq, vec_seq, audio, spectrogram, aux_info

    def set_lang_model(self, lang_model):
        self.lang_model = lang_model

    def _make_speaker_model(self, lmdb_dir, cache_path):
        logging.info('  building a speaker model...')
        speaker_model = Vocab('vid', insert_default_tokens=False)

        lmdb_env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        txn = lmdb_env.begin(write=False)
        cursor = txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            speaker_model.index_word(vid)

        lmdb_env.close()
        logging.info('    indexed %d videos' % speaker_model.n_words)
        self.speaker_model = speaker_model

        # cache
        with open(cache_path, 'wb') as f:
            pickle.dump(self.speaker_model, f)
