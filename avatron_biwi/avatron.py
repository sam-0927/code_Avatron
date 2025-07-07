import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import numpy as np
import copy
import math
import pdb
import numpy as np
import torch.nn.init as init
import random
# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))   #[1/4, 1/16, 1/64, 1/256]
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S, fps=30, hop=267, win=800, sr=16000):    # T:vert. S:mel_frame
    mask = torch.ones(T, S)
    fps_sec = 1/fps
    hop_sec = hop/sr
    win_sec = win/sr
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.transpose(*self.shape)

class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class Conformer(nn.Module):
    def __init__(self,
                 d_model,   # 256
                 nhead,     # 4
                 dim_feedforward=1024,  # 1024
                 dropout=0.1):
        super(Conformer, self).__init__()
        self.nhead = nhead
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward1 = nn.Sequential(nn.LayerNorm(d_model),
                                          Linear(d_model, dim_feedforward),
                                          nn.LeakyReLU(),
                                          nn.Dropout(0.1),
                                          Linear(dim_feedforward, d_model),
                                          nn.Dropout(0.1))

        self.feedforward2 = nn.Sequential(nn.LayerNorm(d_model),
                                          Linear(d_model, dim_feedforward),
                                          nn.LeakyReLU(),
                                          nn.Dropout(0.1),
                                          Linear(dim_feedforward, d_model),
                                          nn.Dropout(0.1))
        self.convolution = nn.Sequential(nn.LayerNorm(d_model),
                                         Transpose(shape=(1,2)),
                                         nn.Conv1d(d_model, d_model*2, kernel_size=5, padding=2),
                                         nn.LeakyReLU(),
                                         nn.Conv1d(d_model*2, d_model, kernel_size=3, padding=1),
                                         nn.Dropout(0.1),
                                         Transpose(shape=(1,2))
                                         )
        self.norm = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
    def forward(self, x, attn_mask=None, padding_mask=None):
        x = 0.5*(x + self.feedforward1(x))
        x = self.norm1(x)
        if attn_mask is not None and attn_mask.size(0) != self.nhead:
            attn_mask = attn_mask.expand(self.nhead, -1, -1)
        x = x.transpose(0,1)
        x_attn, _ = self.multihead_attn(x,x,x,attn_mask=attn_mask)
        x_attn = x_attn.transpose(0,1)
        x = x.transpose(0,1)
              
        x = x + x_attn
        x = x + self.convolution(x)
        x = 0.5*(x + self.feedforward2(x))
        out = self.norm(x)
        
        return out



def linear_interpolation(x, target_frame):  # x:[B,256,seq_len]
    out = F.interpolate(x, size=target_frame, align_corners=True, mode='linear')
    return out.transpose(1,2)

class Avatron(nn.Module):
    def __init__(self, args):
        super(Avatron, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.device = args.device
	    # Preprocessing for TTS feature
        self.pre_conv = nn.Sequential(weight_norm(nn.Conv1d(256, 256, kernel_size=3, padding=1)),
                                      nn.LeakyReLU(),
                                      weight_norm(nn.Conv1d(256, 256, kernel_size=5, padding=2)))
                                      
	    # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        self.conformer = Conformer(d_model=args.feature_dim, nhead=4, dim_feedforward=1024)
        self.lstm = nn.LSTM(args.feature_dim, args.feature_dim//2, 2, bidirectional=True)                                              
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)      
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        self.vertice_dim = args.vertice_dim
        # phoneme label
        self.pho_emb = nn.Embedding(4+1, 256)
        self.pho_fc = nn.Linear(256, 256)
        self.ln = nn.LayerNorm(256)

    def forward(self, up_emb, pho_label, hard_A, template, vertice, one_hot, criterion, up_pho):
        template = template.unsqueeze(1) # (1,1, V*3)
        frame_num = vertice.shape[1]
        
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        
        pho_emb = self.pho_emb(pho_label)
        pho_emb = self.pho_fc(pho_emb)
        up_pho_emb = torch.matmul(hard_A.transpose(1,2), pho_emb)
        if up_pho == 'True':
            gen_input = linear_interpolation((up_emb + up_pho_emb).transpose(1,2), frame_num)
        else:
            gen_input = linear_interpolation((up_emb).transpose(1,2), frame_num)

        gen_input = self.pre_conv(gen_input.transpose(1,2))
        gen_input = gen_input.transpose(1,2)

        gen_input = self.PPE(gen_input + obj_embedding)
        tgt_mask_attn = self.biased_mask[:, :gen_input.shape[1], :gen_input.shape[1]]\
                                    .clone().detach().to(device=self.device)

        gen_out = self.conformer(gen_input, tgt_mask_attn)
        gen_out,_ = self.lstm(gen_out)
        vertice_out = self.vertice_map_r(gen_out)
        vertice = vertice - template

        if torch.isnan(torch.sum(vertice_out)):
            pdb.set_trace()
        
        # Main loss
        loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        loss = torch.mean(loss)
        
        return loss, vertice + template
                
    def predict(self, up_emb, up_pho_emb, template, one_hot, wav_length, up_pho, vertice=None, frame_num=None):
        template = template.unsqueeze(1) # (1,1, V*3)
        ''' hidden_state.shape[1] X 200.xx = wav's sample length
            frame_num = wav's sample length / (sr/fps) '''
        if frame_num is None:
            frame_num = int(wav_length/(16000/25))
        else:
            if vertice is not None:
                vertice = vertice[:, :frame_num]

        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        if up_pho == 'True':
            gen_input = linear_interpolation((up_emb + up_pho_emb).transpose(1,2), frame_num)
        else:
            gen_input = linear_interpolation((up_emb).transpose(1,2), frame_num)
       
        gen_input = self.pre_conv(gen_input.transpose(1,2))
        gen_input = gen_input.transpose(1,2)

        gen_input = self.PPE(gen_input + obj_embedding)
        tgt_mask_attn = self.biased_mask[:, :gen_input.shape[1], :gen_input.shape[1]]\
                                    .clone().detach().to(device=self.device)
        gen_out = self.conformer(gen_input, tgt_mask_attn)
        gen_out,_ = self.lstm(gen_out)

        vertice_out = self.vertice_map_r(gen_out)
        
        vertice_out = vertice_out + template
        
        return vertice_out, vertice

