from pathlib import Path
import shlex
from subprocess import Popen, DEVNULL
from sys import stderr

import numpy as np
import soundfile as sf
import torch

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

    for i in range(n_repeat):
        sample_index = rng.randint(0, n_sample)
        spec = torch.Tensor(dataset[sample_index]['norm_spec'][prior_model.seed_code_len:].copy()).unsqueeze(0)
        seed_motion = torch.Tensor(dataset[sample_index]['keypoints'][:args.prior_seed_len].copy()).unsqueeze(0)
        with torch.no_grad():
            gen_m = prior_model.inference(seed_motion, spec).squeeze().cpu().numpy()
        ori_m = dataset[sample_index]['keypoints'].copy()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        gen_m = dataset.normalized_dir_vec_to_keypoints(gen_m)
        gen_m = dataset.camera_to_world(gen_m)
        poses = {'Original': ori_m, 'Gen': gen_m}
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

    if args.dataset == 'Baijia':
        dataset = BaijiaDataset(args, is_train=False)
    else:
        raise NotImplementedError()

    motion_prior = TransformerPrior(args, is_train=False)
    motion_prior.resume(args.resume_prior)
    motion_prior.net.eval()
    inference(motion_prior, dataset, args)
