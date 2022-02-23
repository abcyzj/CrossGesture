import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import CausalConvolution, ConvNet


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

        B, T, *_ = input_shape
        quantized = quantized.reshape(B, T, self._num_head, self._embedding_dim)
        encoding_indices = encodings.reshape(B, T, self._num_head, self._num_embeddings)

        return loss, quantized, perplexity, encoding_indices

    def lookup_codebook(self, z_motion_one_hot: torch.Tensor):
        """
        :param z_motion_one_hot: (B, T, num_head, num_embedding)
        return: (B, T, num_head, embedding_dim)
        """
        B, T, *_ = z_motion_one_hot.shape
        z_motion_one_hot = z_motion_one_hot.reshape(-1, self._num_head, self._num_embeddings)
        embeddings = torch.einsum('bhn,hnc->bhc', z_motion_one_hot, self._embedding)
        return embeddings.reshape(B, T, self._num_head, self._embedding_dim)


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
            input_channel, args.encoder_channels, args.encoder_downsample_layers, 'downsample', args.norm_type, args.enc_dilations, dropout=args.dropout
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
            return z_specific
        elif self.args.vae_type == 'vqvae':
            spec_output = self.spec_mlp(output)
            commit_loss, quantized, perplexity, encoding_indices = self.vae(spec_output)
            self.commit_loss = commit_loss
            self.perplexity = perplexity.detach() # Only for tensorboard display, do not backward gradient
            return quantized, encoding_indices
        else:
            raise NotImplementedError()

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
            args.num_vq_head * args.pose_hidden_size, args.decoder_channels, args.decoder_upsample_layers, 'upsample', args.norm_type, args.dec_dilations, dropout=args.dropout
        )
        self.pose_g = nn.Sequential(
            nn.Linear(args.decoder_channels[-1], args.decoder_channels[-1]), nn.LeakyReLU(0.1), nn.Linear(args.decoder_channels[-1], output_dim),
        )

    def forward(self, spec_feature: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, H, C) or (B, T, H*C)
        """
        B, T, *_ = spec_feature.shape
        output = spec_feature.reshape(B, T, -1)
        output = self.TCN(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)
        if self.args.joint_repr == 'dir_vec':
            return output
        else:
            raise NotImplementedError()

    def forward_inference(self, motion_code: torch.Tensor): # 废弃了，这样没啥用
        """
        :param motion_code: (B, T, H, C) or (B, T, H*C)
        """
        B, latent_T, *_ = motion_code.shape
        motion_code = motion_code.reshape(B, latent_T, -1)
        num_down_sample_layers = sum([1 if x else 0 for x in self.args.encoder_downsample_layers])
        code_seq_len = self.args.seq_len // 2**num_down_sample_layers
        code_inf_seq_len = self.args.inf_seq_len // 2**num_down_sample_layers
        seq_len = self.args.seq_len
        assert latent_T >= code_seq_len
        output = self.pose_g(self.TCN(motion_code[:, :code_seq_len].permute(0, 2, 1)).permute(0, 2, 1))
        for i in range(code_seq_len//2, code_inf_seq_len - code_seq_len//2, code_seq_len//2):
            output_tail = self.pose_g(self.TCN(motion_code[:, i:i+code_seq_len].permute(0, 2, 1)).permute(0, 2, 1))
            output = torch.cat([output, output_tail[:, seq_len//2:]], dim=1)
        return output


class MelSpecEnc(nn.Module):
    def __init__(self, args):
        """
        :param latent_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__()

        self.convert_dimensions = nn.Conv1d(80, 128, kernel_size=1)
        self.weights_init(self.convert_dimensions)
        self.receptive_field = 1

        convs = []
        norms = []
        conv_len = 3
        for i in range(4):
            dilation = 2**(i % 3)
            self.receptive_field += (conv_len - 1) * dilation
            convs.append(nn.Conv1d(128, 128, kernel_size=conv_len, dilation=dilation))
            norms.append(nn.BatchNorm1d(128))
            self.weights_init(convs[-1])
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.code = nn.Linear(128, args.audio_latent_dim)

        self.apply(lambda x: self.weights_init(x))

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            try:
                nn.init.constant_(m.bias, .01)
            except:
                pass

    def forward(self, spec: torch.Tensor):
        """
        :param spec: (B, T, n_mel, chunk_size)
        :return: code: B x T x latent_dim Tensor containing a latent audio code/embedding
        """
        B, T = spec.shape[0], spec.shape[1]

        # Convert to the right dimensionality
        x = spec.view(-1, spec.shape[2], spec.shape[3])
        x = self.convert_dimensions(x)

        # Process stacks
        for conv, norm in zip(self.convs, self.norms):
            x_ = F.leaky_relu(norm(conv(x)), .2)
            x_ = F.dropout(x_, .2)
            l = (x.shape[2] - x_.shape[2]) // 2
            x = (x[:, :, l:-l] + x_) / 2

        x = torch.mean(x, dim=-1)
        x = x.view(B, T, x.shape[-1])
        x = self.code(x)

        return x


class PriorDec(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        num_head = args.num_vq_head
        num_embedding = args.num_embedding
        hidden = args.prior_hidden_dim
        audio_dim = args.audio_latent_dim

        if args.prior_dec_input == 'onehot':
            self.embedding = nn.Conv1d(num_head*num_embedding, hidden, kernel_size=1)
        elif args.prior_dec_input == 'embedding':
            self.embedding = nn.Conv1d(num_head*args.pose_hidden_size, hidden, kernel_size=1)
        self.conv_layers = nn.ModuleList([
            CausalConvolution(ch_in=hidden, ch_out=hidden, audio_dim=audio_dim, kernel_size=3, dilation=1) for _ in range(args.num_prior_dec_layer)
        ])
        self.logits = nn.Conv1d(hidden, num_head*num_embedding, kernel_size=1)

        self.num_head = num_head
        self.num_embedding = num_embedding
        self.hidden = hidden
        self.audio_dim = audio_dim

    def receptive_field(self):
        recep = 1
        for layer in self.conv_layers:
            recep += layer.receptive_field() - 1
        return recep

    def forward(self, motion_code: torch.Tensor, motion_one_hot: torch.Tensor, audio_code: torch.Tensor):
        """
        :param motion_code: (B, num_hed, embedding_dim, T)
        :param motion_one_hot: (B, num_head, num_embedding, T)
        :param audio_code: (B, audio_latent_dim, T)
        :return logits: (B, num_head, num_embedding, T)
        """

        B, _, _, T = motion_one_hot.shape

        if self.args.prior_dec_input == 'onehot':
            x = self.embedding(motion_one_hot.view(B, -1, T).contiguous())
        elif self.args.prior_dec_input == 'embedding':
            x = self.embedding(motion_code.view(B, -1, T).contiguous())
        for conv in self.conv_layers:
            x = conv(x, audio_code)
            x = F.leaky_relu(x, 0.2)

        logits = self.logits(x)
        logits = logits.reshape(B, self.num_head, self.num_embedding, T)
        return logits

    def forward_inference(self, motion_code: torch.Tensor, motion_one_hot: torch.Tensor, audio_code: torch.Tensor):
        """
        :param motion_code: (B, num_head, embedding_dim, T)
        :param motion_one_hot: (B, num_head, num_embedding, T)
        :param audio_code: (B, audio_latent_dim, T + 1)
        :return logits: (B, num_head, num_embedding, T + 1)
        """
        embedding_dim = motion_code.shape[2]
        B, num_head, num_embedding, T = motion_one_hot.shape
        motion_code = torch.cat([motion_code, torch.zeros(B, num_head, embedding_dim, 1, device=motion_code.device)], dim=-1)
        motion_one_hot = torch.cat([motion_one_hot, torch.zeros(B, num_head, num_embedding, 1, device=motion_one_hot.device)], dim=-1)

        if self.args.prior_dec_input == 'onehot':
            x = self.embedding(motion_one_hot.view(B, -1, T + 1))
        elif self.args.prior_dec_input == 'embedding':
            x = self.embedding(motion_code.view(B, -1, T + 1))
        for conv in self.conv_layers:
            x = conv(x, audio_code)
            x = F.leaky_relu(x, 0.2)

        logits = self.logits(x)
        logits = logits.reshape(B, self.num_head, self.num_embedding, T + 1)
        return logits
