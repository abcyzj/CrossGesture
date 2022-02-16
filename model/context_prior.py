import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .model import Model, MotionVAE
from .module import MelSpecEnc, PriorDec


class ContextPrior(Model):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

        self.motion_vae = MotionVAE(args, is_train=False)
        assert args.resume_vae is not None
        self.motion_vae.resume(args.resume_vae)

        self.net = nn.ModuleDict({
            'spec_enc': MelSpecEnc(args),
            'prior_dec': PriorDec(args)
        }).to(self.device)

        if is_train:
            self.optim = self.init_optim(self.net.parameters())

    def train_one_batch(self, motions: torch.Tensor, norm_spec: torch.Tensor):
        with torch.no_grad():
            _, z_motion_one_hot = self.motion_vae.encode_motion(motions)

        z_motion_one_hot = z_motion_one_hot.permute(0, 2, 3, 1).contiguous() # (B, T, num_head, num_embedding) -> (B, num_head, num_embedding, T)
        audio_code = self.net['spec_enc'](norm_spec)
        audio_code = audio_code.permute(0, 2, 1).contiguous() # (B, T, audio_dim) -> (B, audio_dim, T)
        prior_logits = self.net['prior_dec'](z_motion_one_hot, audio_code)

        z_motion_one_hot = z_motion_one_hot.permute(0, 2, 1, 3).contiguous() # (B, num_head, num_embedding, T) -> (B, num_embedding, num_head, T)
        prior_logits = prior_logits.permute(0, 2, 1, 3).contiguous()

        return z_motion_one_hot, prior_logits

    def train(self, dataloader):
        self.ckpt_dir = os.path.join(self.args.run_dir, 'prior_checkpoints')
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

                z_motion_one_hot, prior_logits = self.train_one_batch(motions, norm_spec)
                loss = self.calculate_loss(z_motion_one_hot, prior_logits, batch)

                loss.backward()
                self.optim.step()

                self.global_step += 1

            self.epoch += 1
            if self.epoch % self.args.save_freq == 0:
                self.save(loss.item())

    def calculate_loss(self, z_motion_one_hot, prior_logits, batch):
        """
        :param z_motion_one_hot: (B, num_embedding, num_head, T)
        :param prior_logits: (B, num_embedding, num_head, T)
        """
        loss_dict = {
            'cross_entro/context_prior': F.cross_entropy(prior_logits, z_motion_one_hot)
        }
        self.log(batch, loss_dict)
        loss = torch.stack(list(loss_dict.values())).sum()
        return loss

    def inference(self, spec: torch.Tensor):
        """
        :param audio: (B, T, audio_dim)
        :return: (B, T, num_head, num_embedding)
        """
        self.net.eval()
        ori_device = spec.device
        spec = spec.to(self.device)
        audio_code = self.net['spec_enc'](spec)
        audio_code = audio_code.permute(0, 2, 1).contiguous() # (B, T, audio_dim) -> (B, audio_dim, T)
        latent_T = audio_code.shape[-1]

        z_motion_one_hot = torch.zeros(1, self.args.num_vq_head, self.args.num_embedding, 0, device=self.device)
        for t in range(latent_T):
            logits = self.net['prior_dec'].forward_inference(z_motion_one_hot, audio_code[:, :, :t+1])
            z_motion_one_hot = torch.cat([z_motion_one_hot, torch.zeros(1, self.args.num_vq_head, self.args.num_embedding, 1, device=self.device)], dim=-1)
            one_hot_ind = torch.argmax(logits[:, :, :, -1], dim=2)
            for h in range(self.args.num_vq_head):
                z_motion_one_hot[:, h, one_hot_ind[:, h].item(), -1] = 1

        z_motion_one_hot = z_motion_one_hot.permute(0, 3, 1, 2).contiguous() # # (B, num_head, num_embedding, T) -> (B, T, num_head, num_embedding)
        motions = self.motion_vae.decode_motion_one_hot(z_motion_one_hot)

        return motions.to(ori_device)

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
