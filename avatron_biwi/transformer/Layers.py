import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from transformer.convbranch import ConvBranch

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
import pdb

class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model//2, d_k//2, d_v//2, dropout=dropout)
        self.convbranch = ConvBranch(d_model//2)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)
            
    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input[:, :, :enc_input.size(2)//2], enc_input[:, :, :enc_input.size(2)//2], enc_input[:, :, :enc_input.size(2)//2], mask=slf_attn_mask)
        local_attn = self.convbranch(enc_input[:, :, enc_input.size(2)//2:])
        enc_output = torch.cat((enc_output,local_attn), dim=2)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self,
                 n_mel_channels=40,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convolutions=5):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels,
                         postnet_embedding_dim,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'),

                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size,
                             stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='tanh'),

                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim,
                         n_mel_channels,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'),

                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
