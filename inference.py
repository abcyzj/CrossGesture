from pathlib import Path

import numpy as np
import torch

from dataset import BaijiaDataset
from model.model import MotionVAE
from options import get_parser
from vis import render_animation


def inference(model, dataset, args):
    n_repeat = 10
    n_sample = len(dataset)
    rng = np.random.RandomState(8848)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_repeat):
        sample_index = rng.randint(0, n_sample)
        motions = torch.Tensor(dataset[sample_index]).unsqueeze(0)
        with torch.no_grad():
            recon_m = model.recon_motion(motions).cpu().numpy().squeeze()
            sample_m = model.sample_motion().cpu().numpy().squeeze()
        recon_m = dataset.normalized_dir_vec_to_keypoints(recon_m)
        sample_m = dataset.normalized_dir_vec_to_keypoints(sample_m)
        ori_m = motions.cpu().numpy().squeeze()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        recon_m = dataset.camera_to_world(recon_m)
        sample_m = dataset.camera_to_world(sample_m)
        poses = {'Original': ori_m, 'Recon': recon_m, 'Sample': sample_m}
        render_animation(poses, dataset.skeleton(), dataset.fps(), 3000, dataset.camera()['azimuth'], result_dir.joinpath(f'recon_{i}.mp4').as_posix())

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.dataset == 'Baijia':
        dataset = BaijiaDataset(args, is_train=False)
    else:
        raise NotImplementedError()

    motion_vae = MotionVAE(args, is_train=False)
    motion_vae.resume(args.resume)
    motion_vae.net_G.eval()
    inference(motion_vae, dataset, args)
