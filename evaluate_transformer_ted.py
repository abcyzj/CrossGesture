import warnings
import pickle
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import signal

from data_loader.ted_dataset import SpeechMotionDataset, default_collate_fn
from pose_embed.evaluator import EmbeddingSpaceEvaluator
from utils.vocab_utils import build_vocab
from model.transformer_prior_ted import TransformerPriorTED
from options import get_parser



def evaluate(prior_model, dataloader, args):
    fgd_evaluator = EmbeddingSpaceEvaluator('/home/yezj/Gesture-Generation-from-Trimodal-Context/output/train_h36m_gesture_autoencoder/gesture_autoencoder_checkpoint_best.bin', torch.device('cuda'))

    for data in tqdm(dataloader, 'Evaluating'):
        _, _, in_text_padded, _, target_vec, in_audio, _, _ = data
        seed_motion = target_vec[:, :args.prior_seed_len]
        word_embed = in_text_padded[:, args.prior_seed_len:]
        with torch.no_grad():
            gen_pose = prior_model.inference(seed_motion, in_audio, word_embed)

        B = gen_pose.shape[0]
        fgd_evaluator.push_samples(gen_pose[:, :34].reshape(B, 34, -1).cuda(), target_vec[:, :34].reshape(B, 34, -1).cuda())

    print(f'FGD: {fgd_evaluator.get_scores()}')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if args.inf_seq_len is None:
        args.inf_seq_len = args.seq_len

    dataset = SpeechMotionDataset(args, is_train=False)
    lang_model = build_vocab('words', [], '/home/yezj/Gesture-Generation-from-Trimodal-Context/data/ted_dataset/vocab_cache.pkl', '/home/yezj/Gesture-Generation-from-Trimodal-Context/data/fasttext/crawl-300d-2M-subword.bin', 300)
    dataset.set_lang_model(lang_model)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=default_collate_fn)

    motion_prior = TransformerPriorTED(args, is_train=False)
    motion_prior.resume(args.resume_prior)
    evaluate(motion_prior, dataloader, args)
