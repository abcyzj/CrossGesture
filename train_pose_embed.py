import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import BaijiaDataset
from pose_embed.embedding_net import EmbeddingNet


def vae_loss(real, recons, mu, log_var):
    recons_loss = F.mse_loss(real, recons)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    loss = recons_loss + kld_loss * 1e-3
    return loss, recons_loss, kld_loss


def eval_epoch(test_loader, embed_vae, device):
    losses = []
    recons_losses = []
    kld_losses = []
    with torch.no_grad():
        for seed_k_seq, *_ in test_loader:
            seed_k_seq = seed_k_seq.to(device)
            pose_feat, pose_mu, pose_logvar, out_pose = embed_vae(seed_k_seq)
            loss, recons_loss, kld_loss = vae_loss(seed_k_seq, out_pose, pose_mu, pose_logvar)
            losses.append(loss.item())
            recons_losses.append(recons_loss.item())
            kld_losses.append(kld_loss.item())

    return np.mean(losses), np.mean(recons_losses), np.mean(kld_losses)


def save_checkpoint(model, optimizer, step_cnt, checkpoint_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step_cnt': step_cnt
    }, checkpoint_path)


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def main(opt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    restore_checkpoint = None
    if opt.restore_from:
        restore_checkpoint = load_checkpoint(opt.restore_from)
        print(f'Restore from {opt.restore_from}')

    train_dataset = BaijiaDataset(opt, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

    embed_vae = EmbeddingNet(opt.pose_dim, 64).to(device)
    optimizer = Adam(embed_vae.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    if restore_checkpoint:
        embed_vae.load_state_dict(restore_checkpoint['model_state_dict'])
        optimizer.load_state_dict(restore_checkpoint['optimizer_state_dict'])

    step_cnt = restore_checkpoint['step_cnt'] if restore_checkpoint else 0
    print_every_n_iter = len(train_loader) // 5

    log_dir = Path(opt.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(opt.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_writer = SummaryWriter(opt.log_dir)

    for epoch in range(opt.start_epoch, opt.start_epoch + opt.n_epoch):
        embed_vae.train()
        train_losses = []
        train_recons_losses = []
        train_kld_losses = []

        for data in tqdm(train_loader, f'Epoch {epoch}'):
            step_cnt += 1
            ori_m = data['keypoints'].to(device)
            ori_m = ori_m.reshape(ori_m.shape[0], ori_m.shape[1], -1)

            optimizer.zero_grad()
            pose_feat, pose_mu, pose_logvar, out_pose = embed_vae(ori_m)
            loss, recons_loss, kld_loss = vae_loss(ori_m, out_pose, pose_mu, pose_logvar)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_recons_losses.append(recons_loss.item())
            train_kld_losses.append(kld_loss.item())

            if step_cnt % 100 == 0:
                log_writer.add_scalar('Loss/train', np.mean(train_losses), step_cnt)
                log_writer.add_scalar('Reconstruction loss/train', np.mean(train_recons_losses), step_cnt)
                log_writer.add_scalar('KLD Loss/train', np.mean(train_kld_losses), step_cnt)

        if epoch % opt.save_every_n_epoch == 0:
            save_checkpoint(embed_vae, optimizer, step_cnt, checkpoint_dir.joinpath(f'{epoch}.pth').as_posix())
            print(f'Save checkpoint on epoch {epoch}, step {step_cnt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--n_epoch', type=int, required=True)
    parser.add_argument('--save_every_n_epoch', type=int, default=5)
    parser.add_argument('--restore_from', type=str)
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--prior_seed_len', type=int)
    parser.add_argument('--seq_stride', type=int, default=20)
    parser.add_argument('--encoder_downsample_layers', nargs='+', type=int, default=[1, 0, 1, 0, 1])
    parser.add_argument('--inf_seq_len', type=int)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--pose_dim', type=int, default=30)
    opt = parser.parse_args()

    main(opt)
