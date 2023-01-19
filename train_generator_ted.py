from pathlib import Path

from torch.utils.data import DataLoader

from data_loader.ted_dataset import SpeechMotionDataset, default_collate_fn
from utils.vocab_utils import build_vocab
from model.transformer_generator_ted import TransformerGeneratorTED
from options import get_parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    del args.config
    parser.write_config_file(args, [run_dir.joinpath('prior_config.ini').as_posix()])

    dataset = SpeechMotionDataset(args, is_train=True)
    lang_model = build_vocab('words', [dataset], '/home/yezj/Gesture-Generation-from-Trimodal-Context/data/ted_dataset/vocab_cache.pkl', '/home/yezj/Gesture-Generation-from-Trimodal-Context/data/fasttext/crawl-300d-2M-subword.bin', 300)
    dataset.set_lang_model(lang_model)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=default_collate_fn)
    print(f'Batch per epoch: {len(dataloader)}')
    motion_prior = TransformerGeneratorTED(args, is_train=True)
    if args.resume_prior:
        motion_prior.resume(args.resume_prior)
    motion_prior.train(dataloader)
