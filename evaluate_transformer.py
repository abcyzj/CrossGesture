import warnings
import pickle
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BaijiaDataset
from model.transformer_prior import TransformerPrior
from options import get_parser


def evaluate(prior_model, dataloader, dataset, args):
    maej = []
    mad = []

    for data in tqdm(dataloader, 'Evaluating'):
        spec = data['norm_spec'][:, prior_model.seed_code_len:]
        seed_motion = data['keypoints'][:, :args.prior_seed_len]
        word_embed = data['word_embed'][:, args.prior_seed_len:]
        silence = data['silence'][:, args.prior_seed_len:]
        with torch.no_grad():
            gen_m = prior_model.inference(seed_motion, spec, word_embed, silence).squeeze().cpu().numpy()
        ori_m = data['keypoints'].cpu().numpy().copy()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        gen_m = dataset.normalized_dir_vec_to_keypoints(gen_m)
        gen_m = dataset.camera_to_world(gen_m)

        ori_a = np.gradient(np.gradient(ori_m[:, args.prior_seed_len:], axis=1), axis=1)
        gen_a = np.gradient(np.gradient(gen_m[:, args.prior_seed_len:], axis=1), axis=1)        
        cur_mad = np.linalg.norm(ori_a - gen_a, axis=3)

        cur_maej = np.linalg.norm(ori_m[:, args.prior_seed_len:] - gen_m[:, args.prior_seed_len:], axis=3)
        maej.append(cur_maej)
        mad.append(cur_mad)

    maej = np.concatenate(maej, axis=0)
    mad = np.concatenate(mad, axis=0)
    print(f'MAEJ: {np.mean(maej)}, MAD: {np.mean(mad)}')
    return {
        'MAEJ': np.mean(maej),
        'MAD': np.mean(mad)
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

    motion_prior = TransformerPrior(args, is_train=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result_dict = {}
        for epoch in range(10, 290, 10):
            motion_prior.resume(os.path.join(args.resume_prior, f'epoch{epoch}.pth'))
            cur_res = evaluate(motion_prior, dataloader, dataset, args)
            result_dict[epoch] = cur_res
    with open(os.path.join(args.resume_prior, 'metrics.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
