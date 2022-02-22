import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from transformer.modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.xavier_normal_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) # add head axis

        q, attn = self.attention(q, k, v, mask)

        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(q)
        q += residual
        
        q = self.layer_norm(q)

        return q, attn


class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, is_conv=False, dropout=0.1):
        super().__init__()
        self.is_conv = is_conv
        if is_conv:
            self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=hparams.ff_conv1d_kernel[0], padding=hparams.ff_conv1d_pad[0])
            self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=hparams.ff_conv1d_kernel[1], padding=hparams.ff_conv1d_pad[1])
        else:
            self.w_1 = nn.Linear(d_in, d_hid)
            self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        if self.is_conv:
            output = x.transpose(1, 2)
            output = self.w_2(F.relu(self.w_1(output)))
            output = output.transpose(1, 2)
        else:
            output = self.w_2(F.relu(self.w_1(x)))
        output = self.layer_norm(output + residual)
        return output
