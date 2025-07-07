import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import copy
import time
import math

import hparams as hp
import utils_pas
import pdb
from transformer.SubLayers import MultiHeadAttention1
from monotonic_align import maximum_path, mask_from_lens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()

        self.internal_aligner = InternalAligner()

        self.prosody_emb = ProsodyEmbedding()
        self.pitch_decoder = ProsodyDecoder()
        self.energy_decoder = ProsodyDecoder()
        self.pitch_emb = nn.Linear(1, hp.encoder_hidden)
        self.energy_emb = nn.Linear(1, hp.encoder_hidden)
    def forward(self, text_pro, src_mask, src_len, attn_mask, 
    		mel_len, mel, mel_mask=None, max_len=None):
        #* x: (B, src_max_len, 256), hard_A: (B,src_max_len,mel_max_len)
        soft_A, hard_A, perf_mask = self.internal_aligner(
				    text_pro, mel, src_len,mel_len, src_mask, mel_mask, attn_mask)
    	#* (B, S, T): binary value -> (B, S): int value
        duration_target = torch.sum(hard_A,dim=2)
        return soft_A, hard_A, perf_mask, duration_target
        
    def inference(self, x, log_duration_prediction):
        duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-hp.log_offset), min=0)
        x, mel_len = self.length_regulator(x, duration_rounded, None)
        mel_mask = utils_pas.get_mask_from_lengths(mel_len)

        return x, mel_len, mel_mask


class InternalAligner(nn.Module):
    '''
    input: text (B, src_max_len, 256) / mel (B, mel_max_len, 80)
    output: Soft_alignment, hard_alignment
    src_mask: (B, max_src_len)
    mel_mask: (B. max_mel_len)
    attn_mask: (B, max_src_len, max_mel_len)
    '''
    def __init__(self):
        super(InternalAligner,self).__init__()
        self.text_encoder = nn.Sequential(
            nn.Conv1d(256, 256*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256*2, 80, kernel_size=1, padding=0)
        )
        #self.linear = nn.Linear(256,80)
        self.mel_encoder = nn.Sequential(
            nn.Conv1d(80,80*2,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(80*2,80,kernel_size=3,padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(80,80,kernel_size=1,padding=0)
        )
        self.softmax = nn.Softmax(dim=1)
        #self.post_conv = nn.Conv2d(1,1,kernel_size=(1,3),stride=1, padding=(0,1), dilation=1, bias=True)

    def forward(self, text, mel,src_len, mel_len, src_mask, mel_mask, attn_mask):
        text_emb = self.text_encoder(text.transpose(1,2))
        text_emb = text_emb.transpose(1,2)
        mel_emb = self.mel_encoder(mel.transpose(1,2))
        mel_emb = mel_emb.transpose(1,2)
        text_emb = text_emb.masked_fill(src_mask.unsqueeze(-1),0)
        mel_emb = mel_emb.masked_fill(mel_mask.unsqueeze(-1),0)
        dist_matrix = torch.cdist(text_emb, mel_emb) #* (B, src_max_len, mel_max_len)
        mask_ST = mask_from_lens(dist_matrix, src_len, mel_len)
        mask_ST_rev = mask_ST == 0
        soft_A = self.softmax((-dist_matrix).masked_fill(mask_ST_rev, -np.inf))
        soft_A = soft_A.masked_fill(mask_ST_rev, 0)
        hard_A = maximum_path(soft_A, mask_ST)
        return soft_A, hard_A, mask_ST_rev   #* (B, S, T)


class VariancePredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(VariancePredictor, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.conv1d_1 = nn.Conv1d(self.input_size,
                                      self.filter_size,
                                      kernel_size=self.kernel,
                                      padding=(self.kernel-1)//2)
        self.relu_1 = nn.ReLU()
        self.layer_norm_1 = nn.LayerNorm(self.filter_size)
        self.dropout_1 = nn.Dropout(self.dropout)

        self.conv1d_2 = nn.Conv1d(self.filter_size,
                                      self.filter_size,
                                      kernel_size=self.kernel,
                                      padding=(self.kernel-1)//2)
        self.relu_2 = nn.ReLU()
        self.layer_norm_2 = nn.LayerNorm(self.filter_size)
        self.dropout_2 = nn.Dropout(self.dropout)

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        encoder_output = encoder_output.contiguous().transpose(1, 2)
        out = self.conv1d_1(encoder_output)
        out = out.contiguous().transpose(1, 2)
        out = self.relu_1(out)
        out = self.layer_norm_1(out)
        out = self.dropout_1(out)

        out = out.contiguous().transpose(1, 2)
        out = self.conv1d_2(out)
        out = out.contiguous().transpose(1, 2)
        out = self.relu_2(out)
        out = self.layer_norm_2(out)
        out = self.dropout_2(out)

        out = self.linear_layer(out)
        out = out.squeeze(-1)
        
        if mask is not None:
            out = out.masked_fill(mask, 0.)
        
        return out

class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = utils_pas.pad(output, max_len)
        else:
            output = utils_pas.pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len



class ProsodyEmbedding(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(ProsodyEmbedding, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.linear_layer = nn.Linear(self.conv_output_size, self.conv_output_size)
        #pitch
        self.conv1d_3 = nn.Conv1d(self.input_size,
                                      self.filter_size,
                                      kernel_size=self.kernel,
                                      padding=(self.kernel-1)//2)
        self.relu_3 = nn.ReLU()
        self.layer_norm_3 = nn.LayerNorm(self.filter_size)
        self.dropout_3 = nn.Dropout(self.dropout)

        # energy
        self.conv1d_4 = nn.Conv1d(self.input_size,
                                      self.filter_size,
                                      kernel_size=self.kernel,
                                      padding=(self.kernel-1)//2)
        self.relu_4 = nn.ReLU()
        self.layer_norm_4 = nn.LayerNorm(self.filter_size)
        self.dropout_4 = nn.Dropout(self.dropout)

        # linear
        self.linear_pitch = nn.Linear(self.conv_output_size, 1)
        self.linear_energy = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask, pros_encoder):

        out = pros_encoder(encoder_output, mask)
        
        pitch_emb = out.contiguous().transpose(1, 2)
        pitch_emb = self.conv1d_3(pitch_emb)
        pitch_emb = pitch_emb.contiguous().transpose(1, 2)
        pitch_emb = self.relu_3(pitch_emb)
        pitch_emb = self.layer_norm_3(pitch_emb)
        pitch_emb = self.dropout_3(pitch_emb)

        energy_emb = out.contiguous().transpose(1, 2)
        energy_emb = self.conv1d_4(energy_emb)
        energy_emb = energy_emb.contiguous().transpose(1, 2)
        energy_emb = self.relu_4(energy_emb)
        energy_emb = self.layer_norm_4(energy_emb)
        energy_emb = self.dropout_4(energy_emb)
        
        out = self.linear_layer(out)    # [B, M, 256]
        pitch = F.sigmoid(self.linear_pitch(pitch_emb))
        energy = F.sigmoid(self.linear_energy(energy_emb))
        pitch = pitch.squeeze(-1)
        energy = energy.squeeze(-1)
        
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(2).expand(-1,-1,self.conv_output_size), 0.)
            pitch = pitch.masked_fill(mask, 0.)
            energy = energy.masked_fill(mask, 0.)
        
        return out, pitch, energy


class ProsodyDecoder(nn.Module):
    """ Pitch and Energy Decoder """

    def __init__(self):
        super(ProsodyDecoder, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.linear_layer = nn.Linear(self.conv_output_size, self.conv_output_size)
        #pitch
        self.conv1d = nn.Conv1d(self.input_size,
                                      self.filter_size,
                                      kernel_size=self.kernel,
                                      padding=(self.kernel-1)//2)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(self.filter_size)
        self.dropout = nn.Dropout(self.dropout)

        # linear
        self.linear = nn.Linear(self.conv_output_size, 1)

    def forward(self, out, mask):
        emb = out.contiguous().transpose(1, 2)
        emb = self.conv1d(emb)
        emb = emb.contiguous().transpose(1, 2)
        emb = self.relu(emb)
        emb = self.layer_norm(emb)
        emb = self.dropout(emb)

        emb = F.sigmoid(self.linear(emb))
        emb = emb.squeeze(-1)
        
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(2).expand(-1,-1,self.conv_output_size), 0.)
            emb = emb.masked_fill(mask, 0.)
        return emb


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class DepthwiseConv(nn.Sequential):
    def __init__(self, d_in, d_out, kernel_size, stride=1, padding=0):
        super(DepthwiseConv, self).__init__(
            nn.Conv1d(d_in, d_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=d_in),
            nn.Conv1d(d_in, d_out, kernel_size=1, groups=hp.num_group)
        )
