import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .module import MotionDec, MotionEnc


class Model:
    def __init__(self, args, is_train):
        super().__init__()
        self.args = args
        self.is_train = is_train

        self.device = torch.device(args.device)
        log_dir = os.path.join(args.run_dir, 'log')
        self.logger = SummaryWriter(log_dir)

        if is_train:
            self.global_step = 0
            self.epoch = 0

    def log(self, batch, loss_dict):
        for key in loss_dict:
            self.logger.add_scalar(key, loss_dict[key].item(), self.global_step)
        if batch % self.args.log_freq == 0:
            logging.info(
                f'Name: {self.args.name}, Epoch: {self.epoch}, Batch: {batch}/{self.batch_counts_per_epoch}'
            )
            for key in loss_dict:
                logging.info(f'{key}: {loss_dict[key].item()}')

    def init_optim(self, param):
        if self.args.optim == 'Adam':
            logging.info('Using Adam optimizer')
            logging.info(f'lr: {self.args.lr}')
            return torch.optim.Adam(param, lr=self.args.lr)
        elif self.args.optim == 'AdamW':
            logging.info('Using AdamW optimizer')
            logging.info(f'lr: {self.args.lr}')
            return torch.optim.AdamW(param, lr=self.args.lr)
        elif self.args.optim == 'RMSProp':
            logging.info('Using RMSProp optimizer')
            logging.info(f'lr: {self.args.lr}')
            return torch.optim.RMSprop(param, lr=self.args.lr)
        elif self.args.optim == 'SGD':
            logging.info('Using SGD optimizer')
            logging.info(f'lr: {self.args.lr}, momentum: {self.args.momentum}')
            return torch.optim.SGD(param, lr=self.args.lr, momentum=self.args.momentum,)
        else:
            raise NotImplementedError()


def init(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class MotionProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode_motion(self, motion: torch.Tensor):
        raise NotImplementedError()

    def decode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def calculate_pos(self, motion: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def calculate_joint_speed(self, pos: torch.Tensor) -> torch.Tensor:
        return pos[:, 1:] - pos[:, :-1]


class BaijiaMotionProcessor(MotionProcessor):
    def __init__(self) -> None:
        super().__init__()

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        assert len(motion.shape) == 4 and motion.shape[2] == 10
        B, T = motion.shape[:2]
        motion = motion.reshape(B, T, -1)
        return motion

    def decode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        assert (
            len(motion.shape) == 3
        ), f'Expects an array of size BxTxC, but received {motion.shape}'
        B, T = motion.shape[:2]
        motion = motion.reshape(B, T, 10, 3)
        return motion


class MotionVAE(Model):
    def __init__(self, args, is_train):
        super().__init__(args, is_train)

        if args.dataset == 'Baijia':
            self.motion_processor = BaijiaMotionProcessor()
        else:
            raise NotImplementedError()

        self.net_G = nn.ModuleDict(
            {
                'motion_enc': MotionEnc(args),
                'motion_dec': MotionDec(args)
            }
        ).to(self.device)
        self.net_G.apply(init)

        if is_train:
            self.optimG = self.init_optim(self.net_G.parameters())

    def sampling(self, size=None, mean=None, var=None):
        if mean is not None and var is not None:
            normal = Normal(mean, var)
            z_x = normal.sample((self.args.seq_len,)).permute(1, 0, 2)
        else:
            z_x = torch.randn(size, device=self.device)
        return z_x

    def encode_motion(self, motions: torch.Tensor):
        if self.net_G.training:
            self.net_G.eval()
        motions = motions.to(self.device)
        motions = self.motion_processor.encode_motion(motions)
        z_motion, z_motion_one_hot = self.net_G['motion_enc'](motions)
        return z_motion, z_motion_one_hot

    def decode_motion_one_hot(self, z_motion_one_hot: torch.Tensor):
        """
        :param z_motion_one_hot: (B, T, num_head, num_embedding)
        return: motion
        """
        if self.net_G.training:
            self.net_G.eval()
        B, T, *_ = z_motion_one_hot.shape
        z_motion = self.net_G['motion_enc'].vae.lookup_codebook(z_motion_one_hot).reshape(B, T, -1)
        dec_m = self.net_G['motion_dec'](z_motion)
        dec_m = self.motion_processor.decode_motion(dec_m)
        return dec_m

    def lookup_codebook(self, z_motion_one_hot: torch.Tensor):
        """
        :param z_motion_one_hot: (B, T, num_head, num_embedding)
        return z_motion: (B, T, num_head, embedding_dim)
        """
        return self.net_G['motion_enc'].vae.lookup_codebook(z_motion_one_hot)

    def recon_motion(self, motions: torch.Tensor):
        motions = motions.to(self.device)

        motions = self.motion_processor.encode_motion(motions)

        z_motion, _ = self.net_G['motion_enc'](motions)
        recon_m = self.net_G['motion_dec'](z_motion)

        recon_m = self.motion_processor.decode_motion(recon_m)
        return recon_m

    def sample_motion(self):
        if self.args.vae_type == 'vae':
            num_down_sample_layers = sum([1 if x else 0 for x in self.args.encoder_downsample_layers])
            z_motion = self.sampling([1, self.args.seq_len // 2**num_down_sample_layers, self.args.pose_hidden_size]).to(self.device)
            recon_m = self.net_G['motion_dec'](z_motion)
            recon_m = self.motion_processor.decode_motion(recon_m)
        elif self.args.vae_type == 'vqvae':
            num_down_sample_layers = sum([1 if x else 0 for x in self.args.encoder_downsample_layers])
            random_indexes = torch.randint(self.args.num_embedding, size=[self.args.inf_seq_len // 2**num_down_sample_layers, self.args.num_vq_head, 1], device=self.device)
            random_encodings = torch.zeros(random_indexes.shape[0], self.args.num_vq_head, self.args.num_embedding, device=self.device)
            random_encodings.scatter_(2, random_indexes, 1)
            z_motion = torch.einsum('bhn,hnc->bhc', random_encodings, self.net_G['motion_enc'].vae._embedding).unsqueeze(0)
            z_motion = z_motion.reshape(z_motion.shape[0], z_motion.shape[1], -1)
            recon_m = self.net_G['motion_dec'](z_motion)
            recon_m = self.motion_processor.decode_motion(recon_m)
        else:
            raise NotImplementedError()
        return recon_m

    def train_one_batch(self, motions: torch.Tensor):
        z_motion_specific, _ = self.net_G['motion_enc'](motions)

        recon_m = self.net_G['motion_dec'](z_motion_specific)

        return recon_m

    def calculate_dir_vec_loss(self, tgt_dir, recon_dir, batch):
        loss_G_dict = {
            'pos/recon_position': F.l1_loss(recon_dir, tgt_dir) * self.args.lambda_pose
        }

        loss_G_dict.update(self.net_G['motion_enc'].get_loss_dict())
        self.log(batch, loss_G_dict)
        loss_G = torch.stack(list(loss_G_dict.values())).sum()
        return loss_G

    def train(self, dataloader):
        ckpt_dir = os.path.join(self.args.run_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.mkdir(ckpt_dir)
        else:
            if len(os.listdir(ckpt_dir)) > 0:
                logging.warning('ckpt dir not empty')
        self.net_G.train()

        self.batch_counts_per_epoch = len(dataloader)

        while self.epoch < self.args.epochs:
            for batch, data in enumerate(tqdm(dataloader, f'Epoch {self.epoch}')):
                motion = data['keypoints'].float().to(self.device)

                self.optimG.zero_grad()

                src_motion = motion.clone()
                tgt_motion = motion.clone()

                recon_m = self.train_one_batch(self.motion_processor.encode_motion(src_motion))
                recon_m = self.motion_processor.decode_motion(recon_m)
                if self.args.joint_repr == 'dir_vec':
                    loss_G = self.calculate_dir_vec_loss(
                        tgt_motion, recon_m, batch
                    )
                else:
                    raise NotImplementedError()

                loss_G.backward()
                self.optimG.step()

                self.global_step += 1

            self.epoch += 1
            if self.epoch % self.args.save_freq == 0:
                self.save(loss_G.item())

    def save(self, loss):
        state = {'args': self.args}
        state['net_G'] = self.net_G.state_dict()
        state['epoch'] = self.epoch
        state['global_step'] = self.global_step
        state['loss'] = loss
        ckpt_dir = os.path.join(self.args.run_dir, 'checkpoints')
        torch.save(state, os.path.join(ckpt_dir, f'epoch{self.epoch}.pth'))
        logging.info(f'parameters of epoch {self.epoch} saved')

    def resume(self, weight_path: str):
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.net_G.load_state_dict(checkpoint['net_G'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.net_G['motion_enc'].set_step(checkpoint['global_step'])
