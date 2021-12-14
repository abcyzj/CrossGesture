import torch
import torch.nn as nn
from torch.nn.functional import linear

from model.layer import ConvNet


class VAE(nn.Module):
    def __init__(self) -> None:
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


class MotionEnc(VAE):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.joint_repr == 'dir_vec':
            joint_repr_dim = 3
        else:
            raise NotImplementedError() # TODO other joint representation
        input_channel = args.joint_num * joint_repr_dim

        conv_mode = 'downsample' if self.args.with_motion_downsample else 'same'
        self.TCN = ConvNet(
            input_channel, args.encoder_channels, args.encoder_downsample_layers, conv_mode, args.norm_type, dropout=args.dropout
        )

        self.spec_linear = nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1])
        self.spec_mean = nn.Sequential(
            nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1]), nn.LeakyReLU(0.1), nn.Linear(args.encoder_channels[-1], args.pose_hidden_size),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(args.encoder_channels[-1], args.encoder_channels[-1]), nn.LeakyReLU(0.1), nn.Linear(args.encoder_channels[-1], args.pose_hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        spec_output = self.spec_linear(output)

        if self.args.with_motion_vae:
            self.z_spec_mu = self.spec_mean(spec_output)
            self.z_spec_var = self.spec_var(spec_output)
            z_specific = self.reparameterize(self.z_spec_mu, self.z_spec_var)
        else:
            z_specific = self.spec_mean(spec_output)

        return z_specific

    def get_loss_dict(self):
        loss_dict = {}
        if self.args.with_motion_vae:
            loss_dict.update(
                {
                    "KL/motion_enc": self.kl_divergence(
                        self.z_spec_mu, self.z_spec_var
                    )
                    * self.args.lambda_kl,
                }
            )
        self.step()
        return loss_dict


class MotionDec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.joint_repr == 'dir_vec':
            joint_repr_dim = 3
        else:
            raise NotImplementedError() # TODO other joint representation
        output_dim = joint_repr_dim * self.args.joint_num

        conv_mode = 'upsample' if self.args.with_motion_downsample else 'same'
        self.TCN = ConvNet(
            args.pose_hidden_size, args.decoder_channels, args.decoder_upsample_layers, conv_mode, args.norm_type, dropout=args.dropout
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
