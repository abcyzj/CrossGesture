from torch import nn


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
        self.bn1 = norm_cls(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        if conv_mode == 'upsample':
            self.up2 = nn.Upsample(scale_factor=2, mode='linear')
        if conv_mode == 'downsample':
            self.conv2 = nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=2,
                padding=padding,
                dilation=dilation
            )
        else:
            self.conv2 = nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation
            )
        self.bn2 = norm_cls(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if conv_mode in ['downsample', 'same']:
            self.net = nn.Sequential(
                self.conv1, self.bn1, self.relu1, self.dropout1,
                self.conv2, self.bn2, self.relu2, self.dropout2
            )
        else:
            self.net = nn.Sequential(
                self.conv1, self.bn1, self.relu1, self.dropout1,
                self.up2, self.bn2, self.relu2, self.dropout2
            )

        if n_inputs == n_outputs and conv_mode == 'same':
            self.downsample = None
        else:
            if conv_mode == 'same':
                self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
            elif conv_mode == 'downsample':
                self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, stride=2)
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
    def __init__(self, num_inputs, num_channels, resample_layers, conv_mode, norm_type, kernel_size=3, dropout=0.2):
        super().__init__()
        assert conv_mode in ['upsample', 'downsample', 'same']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            if conv_mode in ['downsample', 'upsample']:
                layers.append(
                    ResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation_size,
                        padding=((kernel_size - 1)*dilation_size + 1) // 2,
                        conv_mode=conv_mode if resample_layers[i] else 'same',
                        norm_type=norm_type
                    )
                )
            elif conv_mode == 'same':
                layers.append(
                    ResidualBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        dilation=dilation_size,
                        padding=((kernel_size - 1)*dilation_size + 1) // 2,
                        conv_mode=conv_mode,
                        dropout=dropout,
                        norm_type=norm_type
                    )
                )
            else:
                raise NotImplementedError()

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
