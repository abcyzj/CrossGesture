from pathlib import Path

import numpy as np
import torch

from dataset import BaijiaDataset
from model.context_prior import ContextPrior
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
        spec = torch.Tensor(dataset[sample_index]['norm_spec']).unsqueeze(0)
        with torch.no_grad():
            gen_m = prior_model.inference(spec).squeeze().cpu().numpy()
        ori_m = dataset[sample_index]['keypoints'].copy()
        ori_m = dataset.normalized_dir_vec_to_keypoints(ori_m)
        ori_m = dataset.camera_to_world(ori_m)
        gen_m = dataset.normalized_dir_vec_to_keypoints(gen_m)
        gen_m = dataset.camera_to_world(gen_m)
        poses = {'Original': ori_m, 'Gen': gen_m}
        render_animation(poses, dataset.skeleton(), dataset.fps(), 3000, dataset.camera()['azimuth'], result_dir.joinpath(f'recon_{i}.mp4').as_posix())


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.dataset == 'Baijia':
        dataset = BaijiaDataset(args, is_train=False)
    else:
        raise NotImplementedError()

    motion_prior = ContextPrior(args, is_train=False)
    motion_prior.resume(args.resume_prior)
    motion_prior.net.eval()
    inference(motion_prior, dataset, args)
