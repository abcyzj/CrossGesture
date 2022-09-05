import warnings
import pickle
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import signal

from dataset import BaijiaDataset
from pose_embed.evaluator import EmbeddingSpaceEvaluator
from model.transformer_generator import TransformerGenerator
from options import get_parser


def evaluate(prior_model, dataloader, dataset, args):
    fgd_evaluator = EmbeddingSpaceEvaluator('run/pose_embed_v3/checkpoint/100.pth', torch.device('cuda'))
    maej = []
    mad = []
    mvd = []

    for data in tqdm(dataloader, 'Evaluating'):
        spec = data['norm_spec'][:, args.prior_seed_len:]
        seed_motion = data['keypoints'][:, :args.prior_seed_len]
        word_embed = data['word_embed'][:, args.prior_seed_len:]
        silence = data['silence'][:, args.prior_seed_len:]
        with torch.no_grad():
            gen_m = prior_model.inference(seed_motion, spec, word_embed, silence)

        # butter_b, butter_a = signal.butter(10, 3, btype='low', analog=False, output='ba', fs=25)
        # smooth_m = signal.filtfilt(butter_b, butter_a, gen_m, axis=1, padlen=None)
        # gen_m = torch.from_numpy(smooth_m.copy()).float().to(gen_m.device)

        for j in range(2):
            out_pose = gen_m.reshape(gen_m.shape[0], gen_m.shape[1], -1).to(torch.device('cuda'))
            target_k_seq = data['keypoints']
            target_k_seq = target_k_seq.reshape(target_k_seq.shape[0], target_k_seq.shape[1], -1).to(torch.device('cuda'))
            fgd_evaluator.push_samples(out_pose[:, j*64:j*64+64], target_k_seq[:, j*64:j*64+64])
            # fgd_evaluator.push_samples(None, None, torch.cat([out_pose[:, j*34:j*34+34, :3], out_pose[:, j*34:j*34+34, 6:]], dim=2), torch.cat([target_k_seq[:, j*34:j*34+34, :3], target_k_seq[:, j*34:j*34+34, 6:]], dim=2))

        ori_m = data['keypoints'].cpu().numpy()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        gen_m = gen_m.cpu().numpy()
        gen_m = dataset.normalized_dir_vec_to_keypoints(gen_m)
        gen_m = dataset.camera_to_world(gen_m)

        ori_v = np.gradient(ori_m[:, args.prior_seed_len:], axis=1)
        ori_a = np.gradient(ori_v, axis=1)
        gen_v = np.gradient(gen_m[:, args.prior_seed_len:], axis=1)
        gen_a = np.gradient(gen_v, axis=1)
        cur_mvd = np.linalg.norm(ori_v - gen_v, axis=3)
        cur_mad = np.linalg.norm(ori_a - gen_a, axis=3)
        mvd.append(cur_mvd)
        mad.append(cur_mad)

        cur_maej = np.linalg.norm(ori_m[:, args.prior_seed_len:] - gen_m[:, args.prior_seed_len:], axis=3)
        maej.append(cur_maej)
        mad.append(cur_mad)

    maej = np.concatenate(maej, axis=0)
    mad = np.concatenate(mad, axis=0)
    mvd = np.concatenate(mvd, axis=0)
    print(f'MAEJ: {np.mean(maej)}, MAD: {np.mean(mad)}, MVD: {np.mean(mvd)}, FGD: {fgd_evaluator.get_scores()}')
    return {
        'MAEJ': np.mean(maej),
        'MAD': np.mean(mad),
        'FGD': fgd_evaluator.get_scores()
    }

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.inf_seq_len is None:
        args.inf_seq_len = args.seq_len

    if args.dataset == 'Baijia':
        dataset = BaijiaDataset(args, is_train=False)
    else:
        raise NotImplementedError()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    motion_prior = TransformerGenerator(args, is_train=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result_dict = {}
        for epoch in range(200, 210, 10):
            print(f'Epoch {epoch}')
            motion_prior.resume(os.path.join(args.resume_prior, f'epoch{epoch}.pth'))
            cur_res = evaluate(motion_prior, dataloader, dataset, args)
            result_dict[epoch] = cur_res
    with open(os.path.join(args.resume_prior, 'metrics.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
