import torch.nn as nn

from transformer.sublayers import FeedForward, MultiHeadAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, n_head, d_k, d_v, is_conv=True, dropout=0.1):
        super().__init__()
        self.slf_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        self.ffn = FeedForward(d_model, d_hid, is_conv=is_conv, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attention(enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output = self.ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hid, n_head, d_k, d_v, is_conv=False, dropout=0.1):
        super().__init__()
        self.slf_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.ffn = FeedForward(d_model, d_hid, is_conv=is_conv, dropout=dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attention(dec_input, dec_input, dec_input, mask=self_attn_mask)
        dec_output, dec_enc_attn = self.enc_attention(dec_input, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
