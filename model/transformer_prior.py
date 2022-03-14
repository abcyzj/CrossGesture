import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformer.models import Generator

from model.module import MelSpecEnc

from .model import Model, MotionVAE


class TransformerPrior(Model):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

        self.motion_vae = MotionVAE(args, is_train=False)
        assert args.resume_vae is not None
        self.motion_vae.resume(args.resume_vae)
        self.ckpt_dir = os.path.join(self.args.run_dir, 'prior_checkpoints')

        num_downsample_layers = sum([1 if x else 0 for x in args.encoder_downsample_layers])
        self.seed_code_len = args.prior_seed_len // 2**num_downsample_layers
        self.tgt_code_len = args.seq_len // 2**num_downsample_layers
        d_k = args.prior_d_model // args.prior_n_head
        d_v = args.prior_d_model // args.prior_n_head
        self.net = nn.ModuleDict({
            'seed_embed': nn.Linear(args.num_embedding*args.num_vq_head, args.prior_d_model, bias=False),
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
                self.seed_code_len,
                self.tgt_code_len
                ),
            'output': nn.Linear(args.prior_d_model, args.num_embedding*args.num_vq_head),
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

                tgt_code_one_hot, prior_logits = self.train_one_batch(motions, norm_spec, word_embedding, silence)
                loss = self.calculate_loss(tgt_code_one_hot, prior_logits, batch)

                loss.backward()
                self.optim.step()

                self.global_step += 1

            self.epoch += 1
            if self.epoch % self.args.save_freq == 0:
                self.save(loss.item())

    def train_one_batch(self, motions: torch.Tensor, norm_spec: torch.Tensor, word_embedding: torch.Tensor, silence: torch.Tensor):
        with torch.no_grad():
            _, z_motion_one_hot = self.motion_vae.encode_motion(motions)

        B, *_ = motions.shape
        seed_code_one_hot = z_motion_one_hot[:, :self.seed_code_len]
        tgt_code_one_hot = z_motion_one_hot[:, self.seed_code_len:]
        audio_code = self.net['spec_enc'](norm_spec[:, self.seed_code_len:])
        seed_code = self.net['seed_embed'](seed_code_one_hot.reshape(B, self.seed_code_len, -1))
        word_embedding = word_embedding[:, self.args.prior_seed_len:]
        silence = silence[:, self.args.prior_seed_len:]
        prior_output = self.net['gen'](seed_code, audio_code, word_embedding, silence)
        prior_logits = self.net['output'](prior_output).reshape(B, self.tgt_code_len, self.args.num_vq_head, self.args.num_embedding)

        return tgt_code_one_hot, prior_logits

    def calculate_loss(self, tgt_code_one_hot: torch.Tensor, prior_logits: torch.Tensor, batch: int):
        """
        :param tgt_code_one_hot: (B, T, num_head, num_embedding)
        :param prior_logits: (B, T, num_head, num_embedding)
        """
        loss_dict = {
            'cross_entro/context_prior': F.cross_entropy(prior_logits.permute(0, 3, 1, 2), tgt_code_one_hot.permute(0, 3, 1, 2))
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
        _, seed_motion_one_hot = self.motion_vae.encode_motion(seed_motion)
        z_motion_one_hot = [seed_motion_one_hot]
        for i in range(self.args.inf_seq_len // self.args.seq_len):
            audio_code = self.net['spec_enc'](norm_spec[:, i*self.tgt_code_len:(i+1)*self.tgt_code_len])
            seed_code = self.net['seed_embed'](seed_motion_one_hot.reshape(B, self.seed_code_len, -1))
            prior_output = self.net['gen'](seed_code, audio_code, word_embedding[:, i*self.args.seq_len:(i+1)*self.args.seq_len], silence[:, i*self.args.seq_len:(i+1)*self.args.seq_len])
            prior_logits = self.net['output'](prior_output).reshape(B, self.tgt_code_len, self.args.num_vq_head, self.args.num_embedding)

            logprobs = F.log_softmax(prior_logits, dim=3)
            if self.args.prior_inf_sample:
                g = -torch.log(-torch.log(torch.clamp(torch.rand(logprobs.shape, device=logprobs.device), min=1e-10, max=1)))
                logprobs += g
            one_hot_ind = torch.argmax(logprobs, dim=3, keepdim=True)
            cur_z_motion_one_hot = torch.zeros(B, self.tgt_code_len, self.args.num_vq_head, self.args.num_embedding, device=self.device)
            cur_z_motion_one_hot.scatter_(3, one_hot_ind, 1)
            z_motion_one_hot.append(cur_z_motion_one_hot)
            seed_motion_one_hot = cur_z_motion_one_hot[:, -self.seed_code_len:]

        z_motion_one_hot = torch.cat(z_motion_one_hot, dim=1)
        motions = self.motion_vae.decode_motion_one_hot(z_motion_one_hot)
        return motions.to(ori_device)
