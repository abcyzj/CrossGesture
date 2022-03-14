import warnings
import pickle
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import signal

from dataset import BaijiaDataset
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator
from model.model import MotionVAE
from options import get_parser


def evaluate(model, dataloader, dataset, args):
    fgd_evaluator = EmbeddingSpaceEvaluator('data/embedding_net.pth.tar', torch.device('cuda'))
    maej = []
    mad = []

    for data in tqdm(dataloader, 'Evaluating'):
        motions = data['keypoints']
        with torch.no_grad():
            recon_m = model.recon_motion(motions)

        butter_b, butter_a = signal.butter(10, 3, btype='low', analog=False, output='ba', fs=25)
        smooth_m = signal.filtfilt(butter_b, butter_a, recon_m.cpu().numpy(), axis=1, padlen=None)
        recon_m = torch.from_numpy(smooth_m.copy()).float().to(recon_m.device)

        for j in range(0, 128 - 34, 34):
            out_pose = recon_m.reshape(recon_m.shape[0], recon_m.shape[1], -1).to(torch.device('cuda'))
            target_k_seq = data['keypoints']
            target_k_seq = target_k_seq.reshape(target_k_seq.shape[0], target_k_seq.shape[1], -1).to(torch.device('cuda'))
            fgd_evaluator.push_samples(None, None, torch.cat([out_pose[:, j:j+34, :6], out_pose[:, j:j+34, 9:]], dim=2), torch.cat([target_k_seq[:, j:j+34, :6], target_k_seq[:, j:j+34, 9:]], dim=2))

        ori_m = data['keypoints'].cpu().numpy()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        recon_m = recon_m.cpu().numpy()
        recon_m = dataset.normalized_dir_vec_to_keypoints(recon_m)
        recon_m = dataset.camera_to_world(recon_m)

        ori_a = np.gradient(np.gradient(ori_m[:, args.prior_seed_len:], axis=1), axis=1)
        gen_a = np.gradient(np.gradient(recon_m[:, args.prior_seed_len:], axis=1), axis=1)        
        cur_mad = np.linalg.norm(ori_a - gen_a, axis=3)

        cur_maej = np.linalg.norm(ori_m[:, args.prior_seed_len:] - recon_m[:, args.prior_seed_len:], axis=3)
        maej.append(cur_maej)
        mad.append(cur_mad)

    maej = np.concatenate(maej, axis=0)
    mad = np.concatenate(mad, axis=0)
    print(f'MAEJ: {np.mean(maej)}, MAD: {np.mean(mad)}, FGD: {fgd_evaluator.get_scores()}')
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

    motion_vae = MotionVAE(args, is_train=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result_dict = {}
        for epoch in range(130, 170, 10):  
            motion_vae.resume(os.path.join(args.resume_vae, f'epoch{epoch}.pth'))
            print(f'Epoch {epoch}')
            cur_res = evaluate(motion_vae, dataloader, dataset, args)
            result_dict[epoch] = cur_res
    with open(os.path.join(args.resume_vae, 'metrics.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
