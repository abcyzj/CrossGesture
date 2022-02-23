import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, dilation, padding, conv_mode, norm_type, dropout=0.2
    ):
        super().__init__()
        assert conv_mode in ['same', 'upsample', 'downsample']
        assert norm_type in ['batch', 'instance']

        if norm_type == 'batch':
            norm_cls = nn.BatchNorm1d
        else:
            norm_cls = nn.InstanceNorm1d

        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )
        self.norm1 = norm_cls(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        if conv_mode == 'upsample':
            self.up2 = nn.Upsample(scale_factor=2, mode='linear')
        if conv_mode == 'downsample':
            self.down2 = nn.AvgPool1d(2)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )

        self.norm2 = norm_cls(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if conv_mode == 'same':
            self.net = nn.Sequential(
                self.conv1, self.norm1, self.relu1, self.dropout1,
                self.conv2, self.norm2, self.relu2, self.dropout2
            )
        elif conv_mode == 'upsample':
            self.net = nn.Sequential(
                self.conv1, self.norm1, self.relu1, self.dropout1,
                self.up2, self.norm2, self.relu2, self.dropout2
            )
        elif conv_mode == 'downsample':
            self.net = nn.Sequential(
                self.conv1, self.norm2, self.relu1, self.dropout1,
                self.down2, self.norm2, self.relu2, self.dropout2
            )
        else:
            raise NotImplementedError()

        if n_inputs == n_outputs and conv_mode == 'same':
            self.downsample = None
        else:
            if conv_mode == 'same':
                self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
            elif conv_mode == 'downsample':
                self.downsample = nn.Sequential(
                    nn.AvgPool1d(2),
                    nn.Conv1d(n_inputs, n_outputs, 1)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear'),
                    nn.Conv1d(n_inputs, n_outputs, 1)
                )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, resample_layers, conv_mode, norm_type, dilations, kernel_size=3, dropout=0.2):
        super().__init__()
        assert conv_mode in ['upsample', 'downsample']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(
                ResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation_size,
                    padding=((kernel_size - 1)*dilation_size + 1) // 2,
                    conv_mode=conv_mode if resample_layers[i] else 'same',
                    dropout=dropout,
                    norm_type=norm_type
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalConvolution(nn.Module):
    def __init__(self, ch_in, ch_out, audio_dim, kernel_size, dilation):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.audio_dim = audio_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.audio_linear = nn.Conv1d(audio_dim, ch_out, kernel_size=1)

        self.causal = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.InstanceNorm1d(ch_out, track_running_stats=True, affine=True)
        self.dropout = nn.Dropout(0.1)

        self.reset()

    def receptive_field(self):
        return self.dilation * (self.kernel_size - 1) + 1

    def reset(self):
        self.inf_buffer = torch.zeros(1, self.ch_out, 0)

    def forward(self, context: torch.Tensor, audio: torch.Tensor = None):
        """
        :param context: (B, ch_in, T)
        :param audio: (B, audio_dim, T)
        :return (B, ch_out, T)
        """
        h = F.pad(context[:, :, :-1], [self.receptive_field(), 0])
        y = self.causal(h)

        y = self.norm(y) # direct
        y = self.dropout(y)

        if audio is not None:
            audio_latent = self.audio_linear(audio)
            y += audio_latent

        return y

    def forward_inference(self, context: torch.Tensor, audio: torch.Tensor = None):
        """
        :param context: (1, ch_in, T)
        :param audio: (1, audio_dim, T)
        :return: (1, ch_out, T)
        """
        self.inf_buffer = self.inf_buffer.to(context.device)
        self.inf_buffer = torch.cat([self.inf_buffer, torch.zeros(1, self.inf_buffer.shape[1], 1, device=self.inf_buffer.device)], dim=2)

        h = context[:, :, -self.receptive_field()-1:-1]
        if h.shape[-1] < self.receptive_field():
            h = F.pad(h, [self.receptive_field()-h.shape[-1], 0])
        h = self.causal(h)
        self.inf_buffer[:, :, -1:] += h

        if audio is not None:
            audio = audio[:, :, -1:]
            audio_latent = self.audio_linear(audio)
            self.inf_buffer[:, :, -1:] += audio_latent

        return self.inf_buffer
