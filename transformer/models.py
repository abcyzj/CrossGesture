import numpy as np
import torch
import torch.nn as nn

from transformer.layers import DecoderLayer, EncoderLayer


def get_sequence_mask(seq_len):
    sequence_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return sequence_mask

class PositionalEncoding(nn.Module):
    def __init__(self, n_position, d_model):
        super().__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model))

    @staticmethod
    def _get_sinusoid_encoding_table(n_position, d_model):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, d_param, d_model, n_layers, n_head, d_k, d_v, d_inner, is_conv=True, dropout=0.1, n_position=100):
        super().__init__()

        self.src_embed = nn.Linear(d_param, d_model)
        self.position_enc = PositionalEncoding(n_position, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, is_conv=is_conv, dropout=dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, src_seq, self_attn_mask, return_attn=False):
        slf_attn_list = []

        enc_output = self.dropout(self.position_enc(self.src_embed(src_seq)))
        enc_output = self.layer_norm(enc_output)

        for layer in self.layers:
            enc_output, slf_attn = layer(enc_output, self_attn_mask=self_attn_mask)
            slf_attn_list.append(slf_attn)

        if return_attn:
            return enc_output, slf_attn_list
        else:
            return enc_output


class KeypointDecoder(nn.Module):
    def __init__(self, d_input, n_velocity_bin, d_model, n_layers, n_head, d_k, d_v, d_inner, is_conv=False, dropout=0.1, n_position=100):
        super().__init__()

        self.d_input = d_input
        self.input_embed_layer = nn.Embedding(d_input * n_velocity_bin, d_model)
        self.position_enc = PositionalEncoding(n_position, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, is_conv=is_conv, dropout=dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, dec_input, self_attn_mask, enc_output, dec_enc_attn_mask, return_attn=False):
        slf_attn_list, dec_enc_attn_list = [], []

        input_embed = self.input_embed_layer(dec_input)
        input_embed = torch.sum(input_embed, dim=-2)
        dec_output = self.dropout(self.position_enc(input_embed))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layers:
            dec_output, slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, self_attn_mask=self_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask
            )
            if return_attn:
                slf_attn_list.append(slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)

        if return_attn:
            return dec_output, slf_attn_list, dec_enc_attn_list
        else:
            return dec_output


class Decoder(nn.Module):
    def __init__(self, d_mel, d_word, d_model, n_layers, downsample_layers, n_head, d_k, d_v, d_inner, is_conv=False, dropout=0.1, n_position=100):
        super().__init__()

        self.mel_embed_layer = nn.Linear(d_mel, d_model)
        self.mel_pos_enc = PositionalEncoding(n_position, d_model)
        self.mel_dropout = nn.Dropout(dropout)
        self.mel_layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, is_conv=is_conv, dropout=dropout) for _ in range(n_layers)
            ])
        self.mel_layer_norm = nn.LayerNorm(d_model)

        self.silence_embed = nn.Linear(1, d_model, bias=False)
        self.word_embed_layer = nn.Linear(d_word, d_model)
        num_downsample_layer = sum([1 if x else 0 for x in downsample_layers])
        self.word_pos_enc = PositionalEncoding(n_position*2**num_downsample_layer, d_model)
        self.word_dropout = nn.Dropout(dropout)
        self.word_layers = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, downsample=downsample_layers[i], is_conv=is_conv, dropout=dropout) for i in range(n_layers)
        ])
        self.word_layer_norm = nn.LayerNorm(d_model)

        self.position_enc = PositionalEncoding(n_position, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, is_conv=is_conv, dropout=dropout) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, melspec, word_embedding, silence, self_attn_mask, enc_output, dec_enc_attn_mask, return_attn=False):
        slf_attn_list, dec_enc_attn_list = [], []

        mel_embed = self.mel_embed_layer(melspec)
        mel_input = self.mel_dropout(self.mel_pos_enc(mel_embed))
        for layer in self.mel_layers:
            mel_input, _ = layer(mel_input, self_attn_mask=self_attn_mask)
        mel_input = self.mel_layer_norm(mel_input)

        word_embed = self.word_embed_layer(word_embedding) + self.silence_embed(silence)
        word_input = self.word_dropout(self.word_pos_enc(word_embed))
        for layer in self.word_layers:
            word_input, _ = layer(word_input, self_attn_mask=self_attn_mask)
        word_input = self.word_layer_norm(word_input)
        dec_output = self.dropout(self.position_enc(mel_input + word_input))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.dec_layers:
            dec_output, slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, self_attn_mask=self_attn_mask, dec_enc_attn_mask=dec_enc_attn_mask
            )
            if return_attn:
                slf_attn_list.append(slf_attn)
                dec_enc_attn_list.append(dec_enc_attn)

        if return_attn:
            return dec_output, slf_attn_list, dec_enc_attn_list
        else:
            return dec_output


class Generator(nn.Module):
    def __init__(self, d_param, d_mel, d_word, d_model, n_layers, downsample_layer, n_head, d_k, d_v, d_inner, seed_len, predict_len, enc_conv=True, dec_conv=True, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(d_param, d_model, n_layers, n_head, d_k, d_v, d_inner, is_conv=enc_conv, dropout=dropout, n_position=seed_len)
        self.decoder = Decoder(d_mel, d_word, d_model, n_layers, downsample_layer, n_head, d_k, d_v, d_inner, is_conv=dec_conv, dropout=dropout, n_position=predict_len)
        self.fc = nn.Linear(d_model, d_param)

    def forward(self, seed_k_seq, melspec, word_embedding, silence):
        enc_output = self.encoder(seed_k_seq, self_attn_mask=None)
        dec_output = self.decoder(melspec, word_embedding, silence, None, enc_output, None)
        output = self.fc(dec_output)
        return output
