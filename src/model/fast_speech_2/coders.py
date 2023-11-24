import torch
from torch import nn
from torch.nn import Sequential
import torch.nn.functional as F
from src.base import BaseModel
from src.model.fast_speech_2.attention import MultiHeadAttention


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(0).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """For masking out the padding part of key sequence."""
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, kernel_size=3, dropout=0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(d_in, d_hid, kernel_size=kernel_size, padding="same")
        # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, kernel_size=kernel_size, padding="same")

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class Encoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, num_heads, filter_size, dropout, 
                 vocab_size, max_seq_len, padding_idx):
        super().__init__()
        self.src_word_emb = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=padding_idx,
        )

        self.position_enc = nn.Embedding(
            max_seq_len + 1, 
            hidden_dim, 
            padding_idx=padding_idx
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    hidden_dim,
                    filter_size,
                    num_heads,
                    hidden_dim // num_heads,
                    hidden_dim // num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, src_pos):
        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )
        return enc_output, non_pad_mask


class Decoder(nn.Module):
    def __init__(self, n_layers, hidden_dim, num_heads, filter_size, dropout, 
                 vocab_size, padding_idx):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.src_word_emb = nn.Embedding(
            vocab_size,
            hidden_dim,
            padding_idx=padding_idx,
        )

        self.layer_stack = nn.ModuleList(
            [
                FFTBlock(
                    hidden_dim,
                    filter_size,
                    num_heads,
                    hidden_dim // num_heads,
                    hidden_dim // num_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, enc_pos):

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.src_word_emb(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask
            )

        return dec_output
