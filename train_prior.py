from pathlib import Path

from torch.utils.data import DataLoader

from dataset import BaijiaDataset
from model.context_prior import ContextPrior
from options import get_parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    del args.config
    parser.write_config_file(args, [run_dir.joinpath('prior_config.ini').as_posix()])

    if args.dataset == 'Baijia':
        dataset = BaijiaDataset(args, is_train=True)
    else:
        raise NotImplementedError()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(f'Batch per epoch: {len(dataloader)}')
    motion_vae = ContextPrior(args, is_train=True)
    if args.resume_prior:
        motion_vae.resume(args.resume_prior)
    motion_vae.train(dataloader)
