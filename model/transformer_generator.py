import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformer.models import Generator

from model.module import MelSpecEnc

from .model import Model, MotionVAE


class TransformerGenerator(Model):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

        self.ckpt_dir = os.path.join(self.args.run_dir, 'prior_checkpoints')

        d_k = args.prior_d_model // args.prior_n_head
        d_v = args.prior_d_model // args.prior_n_head
        self.net = nn.ModuleDict({
            'seed_embed': nn.Linear(args.joint_num*3, args.prior_d_model),
            'gen': Generator(
                args.prior_d_model,
                args.audio_latent_dim,
                args.prior_d_word,
                args.prior_d_model,
                args.num_prior_dec_layer,
                args.prior_downsample_layer,
                args.prior_n_head,
                d_k,
                d_v,
                args.prior_d_inner,
                args.prior_seed_len,
                args.seq_len
                ),
            'output': nn.Linear(args.prior_d_model, args.joint_num*3),
            'spec_enc': MelSpecEnc(args)
        }).to(self.device)

        if is_train:
            self.optim = self.init_optim(self.net.parameters())

    def train(self, dataloader):
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        else:
            if len(os.listdir(self.ckpt_dir)) > 0:
                logging.warning('ckpt dir not empty')
        self.net.train()

        self.batch_counts_per_epoch = len(dataloader)
        while self.epoch < self.args.epochs:
            for batch, data in enumerate(tqdm(dataloader, f'Epoch {self.epoch}')):
                motions = data['keypoints'].float().to(self.device)
                norm_spec = data['norm_spec'].float().to(self.device)
                word_embedding = data['word_embed'].float().to(self.device)
                silence = data['silence'].float().to(self.device)

                self.optim.zero_grad()

                out_pose = self.train_one_batch(motions, norm_spec, word_embedding, silence)
                loss = self.calculate_loss(motions[:, self.args.prior_seed_len:],out_pose, batch)

                loss.backward()
                self.optim.step()

                self.global_step += 1

            self.epoch += 1
            if self.epoch % self.args.save_freq == 0:
                self.save(loss.item())

    def train_one_batch(self, motions: torch.Tensor, norm_spec: torch.Tensor, word_embedding: torch.Tensor, silence: torch.Tensor):
        B, *_ = motions.shape
        audio_code = self.net['spec_enc'](norm_spec[:, self.args.prior_seed_len:])
        seed_code = self.net['seed_embed'](motions[:, :self.args.prior_seed_len].reshape(B, self.args.prior_seed_len, -1))
        word_embedding = word_embedding[:, self.args.prior_seed_len:]
        silence = silence[:, self.args.prior_seed_len:]
        prior_output = self.net['gen'](seed_code, audio_code, word_embedding, silence)
        out_pose = self.net['output'](prior_output).reshape(B, self.args.seq_len, self.args.joint_num, -1)

        return out_pose

    def calculate_loss(self, target_pose: torch.Tensor, out_pose: torch.Tensor, batch: int):
        loss_dict = {
            'cross_entro/context_prior': F.l1_loss(out_pose, target_pose)
        }
        self.log(batch, loss_dict)
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss

    def save(self, loss):
        state = {'args': self.args}
        state['net'] = self.net.state_dict()
        state['epoch'] = self.epoch
        state['global_step'] = self.global_step
        state['loss'] = loss
        torch.save(state, os.path.join(self.ckpt_dir, f'epoch{self.epoch}.pth'))
        logging.info(f'parameters of epoch {self.epoch} saved')

    def resume(self, weight_path):
        state = torch.load(weight_path, map_location=self.device)
        self.net.load_state_dict(state['net'])
        self.epoch = state['epoch']
        self.global_step = state['global_step']

    def inference(self, seed_motion: torch.Tensor, norm_spec: torch.Tensor, word_embedding: torch.Tensor, silence: torch.Tensor):
        """
        :param seed_motion: (B, T_seed, 30)
        :param norm_spec: (B, T_tgt, audio_dim)
        """
        self.net.eval()
        ori_device = seed_motion.device
        seed_motion = seed_motion.float().to(self.device)
        norm_spec = norm_spec.float().to(self.device)
        word_embedding = word_embedding.float().to(self.device)
        silence = silence.float().to(self.device)
        B = seed_motion.shape[0]
        out_motions = [seed_motion]
        for i in range(self.args.inf_seq_len // self.args.seq_len):
            audio_code = self.net['spec_enc'](norm_spec[:, i*self.args.seq_len:(i+1)*self.args.seq_len])
            seed_code = self.net['seed_embed'](seed_motion.reshape(B, self.args.prior_seed_len, -1))
            prior_output = self.net['gen'](seed_code, audio_code, word_embedding[:, i*self.args.seq_len:(i+1)*self.args.seq_len], silence[:, i*self.args.seq_len:(i+1)*self.args.seq_len])
            cur_out_motion = self.net['output'](prior_output).reshape(B, self.args.seq_len, self.args.joint_num, -1)
            out_motions.append(cur_out_motion)
            seed_motion = cur_out_motion[:, -self.args.prior_seed_len:]

        out_motions = torch.cat(out_motions, dim=1)
        return out_motions.to(ori_device)
