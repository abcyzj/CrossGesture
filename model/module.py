import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import ConvNet


class VanillaVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_step = 0

    @staticmethod
    def reparameterize(mu, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu, var):
        return torch.mean(-0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=2))

    def step(self):
        self.global_step += 1


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, num_head, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._num_head = num_head

        self.register_buffer('_embedding', torch.Tensor(self._num_head, self._num_embeddings, self._embedding_dim))
        self._embedding.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(self._num_head, num_embeddings))
        self.register_buffer('_ema_w', torch.Tensor(self._num_head, num_embeddings, self._embedding_dim))
        self._ema_w.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # inputs: the shape of (B, T, C)
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._num_head, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=2, keepdim=True) 
                    + torch.sum(self._embedding**2, dim=2).unsqueeze(0)
                    - 2 * torch.einsum('bhc,hnc->bhn', flat_input, self._embedding))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=2, keepdim=True)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_head, self._num_embeddings, device=inputs.device)
        encodings.scatter_(2, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.einsum('bhn,hnc->bhc', encodings, self._embedding).reshape(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size, dim=1, keepdim=True)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.einsum('bhn,bhc->hnc', encodings, flat_input.detach()) # Note: flat_input requires gradient
            self._ema_w = self._ema_w * self._decay + (1 - self._decay) * dw

            self._embedding = self._ema_w / self._ema_cluster_size.unsqueeze(2)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=(0, 1))
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class MotionEnc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.joint_repr == 'dir_vec':
            joint_repr_dim = 3
        else:
            raise NotImplementedError() # TODO other joint representation
        input_channel = args.joint_num * joint_repr_dim

        self.TCN = ConvNet(
            input_channel, args.encoder_channels, args.encoder_downsample_layers, 'downsample', args.norm_type, dropout=args.dropout
        )

        if args.vae_type == 'vae':
            self.spec_linear = nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1])
            self.spec_mean = nn.Sequential(
                nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1]),
                nn.LeakyReLU(0.1),
                nn.Linear(args.encoder_channels[-1], args.pose_hidden_size)
            )
            self.spec_var = nn.Sequential(
                nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1]),
                nn.LeakyReLU(0.1),
                nn.Linear(args.encoder_channels[-1], args.pose_hidden_size)
            )
            self.vae = VanillaVAE()
        elif args.vae_type == 'vqvae':
            self.spec_mlp = nn.Sequential(
                nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1]),
                nn.LeakyReLU(0.1),
                nn.Linear(args.encoder_channels[-1], args.num_vq_head * args.pose_hidden_size)
            )
            self.vae = VectorQuantizerEMA(args.num_embedding, args.num_vq_head, args.pose_hidden_size, args.lambda_commit, args.lambda_ema_decay)

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        if self.args.vae_type == 'vae':
            spec_output = self.spec_linear(output)
            self.z_spec_mu = self.spec_mean(spec_output)
            self.z_spec_var = self.spec_var(spec_output)
            z_specific = self.vae.reparameterize(self.z_spec_mu, self.z_spec_var)
        elif self.args.vae_type == 'vqvae':
            spec_output = self.spec_mlp(output)
            commit_loss, quantized, perplexity, _ = self.vae(spec_output)
            self.commit_loss = commit_loss
            self.perplexity = perplexity.detach() # Only for tensorboard display, do not backward gradient
            z_specific = quantized
        else:
            raise NotImplementedError()

        return z_specific

    def get_loss_dict(self):
        loss_dict = {}
        if self.args.vae_type == 'vae':
            loss_dict.update(
                {
                    'KL/motion_enc': self.vae.kl_divergence(
                        self.z_spec_mu, self.z_spec_var
                    ) * self.args.lambda_kl
                }
            )
            self.vae.step()
        elif self.args.vae_type == 'vqvae':
            loss_dict.update(
                {
                    'commit_loss/motion_enc': self.commit_loss,
                    'perplexity/motion_enc': self.perplexity
                }
            )
        return loss_dict

    def set_step(self, step):
        if self.args.vae_type == 'vae':
            self.vae.global_step = step


class MotionDec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.joint_repr == 'dir_vec':
            joint_repr_dim = 3
        else:
            raise NotImplementedError() # TODO other joint representation
        output_dim = joint_repr_dim * self.args.joint_num

        self.TCN = ConvNet(
            args.num_vq_head * args.pose_hidden_size, args.decoder_channels, args.decoder_upsample_layers, 'upsample', args.norm_type, dropout=args.dropout
        )
        self.pose_g = nn.Sequential(
            nn.Linear(args.decoder_channels[-1], args.decoder_channels[-1]), nn.LeakyReLU(0.1), nn.Linear(args.decoder_channels[-1], output_dim),
        )

    def forward(self, spec_feature: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = spec_feature
        output = self.TCN(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)
        if self.args.joint_repr == 'dir_vec':
            return output
        else:
            raise NotImplementedError()
