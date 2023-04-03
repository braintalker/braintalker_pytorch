import torch
import torch.nn as nn
from torch.nn import functional as F
from models.SubLayers import MultiHeadAttention, PositionwiseFeedForward


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,
                 dropout,
                 **kwargs):


        super(FFTBlock, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel,fft_conv1d_padding, dropout=dropout)
        self.fc = nn.Linear(d_model,1)     

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None, return_attn=False):

        # enc_input = [B,T,C]
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        if return_attn:
            return enc_slf_attn
        else:
            enc_output = self.pos_ffn(enc_output)

            return enc_output



class Decoder(nn.Module):

    def __init__(self,
                 in_channel,
                 d_model,
                 d_inner,
                 n_head,
                 n_layers,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,             
                 dropout,
                 **kwargs):

        super(Decoder, self).__init__()
        d_k = d_model // n_head
        d_v = d_model // n_head
        self.slf_attn = MultiHeadAttention
        self.fc = nn.Linear(d_model,1)   
        self.conv = nn.Conv1d(in_channel, d_model, kernel_size=3, padding=1)
        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, fft_conv1d_kernel,fft_conv1d_padding, dropout) for _ in range(n_layers)])
        n_head = 1
        self.channel_attention = nn.Linear(64, 64)

    def forward(self, enc, non_pad_mask=None, slf_attn_mask=None):

        enc_output = enc
        enc_output = self.conv(enc_output.transpose(1,2))
        enc_output = enc_output.transpose(1,2)

        output = enc_output
        
        for enc_layer in self.layer_stack:
            output = enc_layer(
                output)

        return output
