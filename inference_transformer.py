import os
from pathlib import Path
import shlex
from subprocess import Popen, DEVNULL
from sys import stderr
import warnings

import numpy as np
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal

from dataset import BaijiaDataset
from model.transformer_prior import TransformerPrior
from options import get_parser
from vis import render_animation


def inference(prior_model, dataset, args):
    n_repeat = 10
    n_sample = len(dataset)
    rng = np.random.RandomState(8848)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    l_scores = []
    r_scores = []

    for i in tqdm(range(n_repeat)):
        sample_index = rng.randint(0, n_sample)
        spec = torch.Tensor(dataset[sample_index]['norm_spec'][prior_model.seed_code_len:].copy()).unsqueeze(0)
        seed_motion = torch.Tensor(dataset[sample_index]['keypoints'][:args.prior_seed_len].copy()).unsqueeze(0)
        word_embed = torch.Tensor(dataset[sample_index]['word_embed'][args.prior_seed_len:].copy()).unsqueeze(0)
        silence = torch.Tensor(dataset[sample_index]['silence'][args.prior_seed_len:].copy()).unsqueeze(0)
        with torch.no_grad():
            gen_m = prior_model.inference(seed_motion, spec, word_embed, silence).squeeze().cpu().numpy()
        ori_m = dataset[sample_index]['keypoints'].copy()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        gen_m = dataset.normalized_dir_vec_to_keypoints(gen_m)
        gen_m = dataset.camera_to_world(gen_m)

    # Calculate SMS
    #     ori_v = np.gradient(ori_m[args.prior_seed_len:], axis=0)
    #     ori_v_max_ind = np.argmax(np.linalg.norm(ori_v, axis=2), axis=0)
    #     gen_v = np.gradient(gen_m[args.prior_seed_len:], axis=0)
    #     w = 16
    #     j_ind = 7
    #     l = ori_v_max_ind[j_ind] // w
    #     r = l + w
    #     score = np.dot(gen_v[l:r, j_ind], ori_v[ori_v_max_ind[j_ind], j_ind]) / np.linalg.norm(ori_v[ori_v_max_ind[j_ind], j_ind])**2
    #     l_scores.append(score.max())
    #     j_ind = 10
    #     l = ori_v_max_ind[j_ind] // w
    #     r = l + w
    #     score = np.dot(gen_v[l:r, j_ind], ori_v[ori_v_max_ind[j_ind], j_ind]) / np.linalg.norm(ori_v[ori_v_max_ind[j_ind], j_ind])**2
    #     r_scores.append(score.max())

    # print(f'Mean score: {np.mean(r_scores)}, {np.mean(l_scores)}')

        # poses = {'Original': ori_m, 'Gen': gen_m}
        butter_b, butter_a = signal.butter(10, 3, btype='low', analog=False, output='ba', fs=25)
        gen_m = signal.filtfilt(butter_b, butter_a, gen_m, axis=0, padlen=None)
        poses = {'Pose': gen_m}
        tmp_video_f = result_dir.joinpath(f'recon_{i}_tmp.mp4')
        tmp_audio_f = result_dir.joinpath(f'recon_{i}.wav')
        final_video_f = result_dir.joinpath(f'recon_{i}.mp4')
        render_animation(poses, dataset.skeleton(), dataset.fps(), 3000, dataset.camera()['azimuth'], tmp_video_f.as_posix())
        audio = dataset[sample_index]['ori_audio']
        sf.write(tmp_audio_f.as_posix(), audio, dataset[sample_index]['ori_sr'])
        p = Popen(
            shlex.split(f'ffmpeg -y -i {tmp_video_f.as_posix()} -i {tmp_audio_f.as_posix()} -map 0 -map 1:a -c:v copy -shortest {final_video_f.as_posix()}'),
            stdout=DEVNULL,
            stderr=DEVNULL
        )
        p.wait()
        tmp_video_f.unlink()
        tmp_audio_f.unlink()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.inf_seq_len is None:
        args.inf_seq_len = args.seq_len

    if args.dataset == 'Baijia':
        dataset = BaijiaDataset(args, is_train=False)
    else:
        raise NotImplementedError()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        motion_prior = TransformerPrior(args, is_train=False)
        for epoch in range(200, 210, 10):
            motion_prior.resume(os.path.join(args.resume_prior, f'epoch{epoch}.pth'))
            motion_prior.net.eval()
            print(epoch)
            inference(motion_prior, dataset, args)
