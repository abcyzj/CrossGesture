import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model.module import MelSpecEnc

from transformer.models import Generator
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
        d_k = args.pose_hidden_size // args.num_prior_dec_head
        d_v = args.pose_hidden_size // args.num_prior_dec_head
        self.net = nn.ModuleDict({
            'input_embed': nn.Linear(args.num_embedding*args.num_vq_head, args.pose_hidden_size, bias=False),
            'gen': Generator(
                args.pose_hidden_size,
                args.audio_latent_dim,
                args.pose_hidden_size,
                args.num_prior_dec_layer,
                args.prior_n_head,
                d_k,
                d_v,
                args.prior_d_inner,
                self.seed_code_len,
                self.tgt_code_len
                ),
            'output': nn.Linear(args.pose_hidden_dim, args.num_embedding*args.num_vq_head),
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
                
                self.optim.zero_grad()

                tgt_code_one_hot, prior_logits = self.train_one_batch(motions, norm_spec)
                loss = self.calculate_loss(tgt_code_one_hot, prior_logits, batch)

                loss.backward()
                self.optim.step()

                self.global_step += 1

            self.epoch += 1
            if self.epoch % self.args.save_freq == 0:
                self.save(loss.item())

    def train_one_batch(self, motions: torch.Tensor, norm_spec: torch.Tensor):
        with torch.no_grad():
            _, z_motion_one_hot = self.motion_vae.encode_motion(motions)
        
        B, T, *_ = motions.shape
        seed_code_one_hot = z_motion_one_hot[:, :self.seed_code_len]
        tgt_code_one_hot = z_motion_one_hot[:, self.seed_code_len:]
        audio_code = self.net['spec_enc'](norm_spec[:, :self.seed_code_len])
        seed_code = self.net['input_embed'](seed_code_one_hot.reshape(B, T, -1))
        prior_output = self.net['gen'](seed_code, audio_code)
        prior_logits = self.net['output'](prior_output).reshape(B, T, self.args.num_vq_head, self.args.num_embedding)

        return tgt_code_one_hot, prior_logits

    def calculate_loss(self, tgt_code_one_hot: torch.Tensor, prior_logits: torch.Tensor, batch: int):
        """
        :param tgt_code_one_hot: (B, T, num_head, num_embedding)
        :param prior_logits: (B, T, num_head, num_embedding)
        """
        loss_dict = {
            'corss_entro/context_prior': F.cross_entropy(prior_logits.permute(0, 3, 1, 2), tgt_code_one_hot.permute(0, 3, 1, 2))
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
