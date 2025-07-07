import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy.io import wavfile
import os

import text
import hparams as hp
import matplotlib.pylab as plab
import json
from env import AttrDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def get_alignment(tier):
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(int(e*hp.sampling_rate/hp.hop_length)-int(s*hp.sampling_rate/hp.hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, durations, start_time, end_time

def process_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        text = []
        name = []
        spk = []
        emo = []
        for line in f.readlines():
            if len(line.strip('\n').split('|')) == 4:
                path, t, s, e = line.strip('\n').split('|')
                emo.append(e)
            else:
                path, t, s = line.strip('\n').split('|')
            n = path.split('/')[-1][:-4]
            name.append(n)
            text.append(t)
            spk.append(s)
        if len(emo) > 0:
            return name, text, spk, emo
        return name, text, spk, None

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def plot_data(data, titles=None, filename=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax, offset=0):
        ax = fig.add_axes(old_ax.get_position(), anchor='W')
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        # spectrogram, pitch, energy = data[i]
        spectrogram = data[i]
        axes[i][0].imshow(spectrogram, origin='lower')
        axes[i][0].set_aspect(2.5, adjustable='box')
        axes[i][0].set_ylim(0, hp.n_mel_channels)
        axes[i][0].set_title(titles[i], fontsize='medium')
        axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False) 
        axes[i][0].set_anchor('W')
        
    plt.savefig(filename, dpi=200)
    plt.close()

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))
    return mask

def get_mask_hard(hard, value=0.5):
    mask = torch.zeros_like(hard).cuda()
    hard_len = torch.sum(hard, dim=2)
    B, L = mask.size(0), mask.size(1)
    for i in range(B):
        for j in range(L):
            idx_list = hard[i][j].nonzero(as_tuple=True)[0]
            if len(idx_list) == 0:
                break
            start = idx_list[0]; end = idx_list[-1]
            if end+1 == torch.sum(hard_len, dim=1)[i]:
                mask[i][j][start-1] = value
                break
            elif start == 0:
                mask[i][j][end+1] = value
            else:
                mask[i][j][start-1] = value
                if end+1 > len(mask[i][j])-1:
                    pdb.set_trace()
                mask[i][j][end+1] = value
    return mask
def get_mask_from_matrix(x):   # [B,L,T]
    mask = torch.zeros_like(x).cuda()
    mask = ( x >= 1e-2)

    return mask
def get_mask_residual(hard):
    mask = torch.zeros_like(hard).cuda()
    
    for i in range(mask.size(0)):
        for j in range(mask.size(1)):
            idx_list = hard[i][j].nonzero(as_tuple=True)[0]
            if len(idx_list) == 0:
                break
            start = idx_list[0]; end = idx_list[-1]
            if start == 0: start = 1
            if end == len(hard[i][j])-1: end = end-1
            mask[i][j][start-1:end+2] = 1
    return mask
def get_mask_from_position(src_seq, max_len=None):
    batch_size = src_seq.shape[0]
    if max_len is None:
        max_len = len(src_seq).item()
    mask = (148 == src_seq)
    return mask

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded

def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output

def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def plot_alignment(alignment, fn):
    # [4, encoder_step, decoder_step] 
    fig, axes = plab.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            g = axes[i][j].imshow(alignment[i*2+j,:,:].T,
                aspect='auto', origin='lower',
                interpolation='none')
            plab.colorbar(g, ax=axes[i][j])
    
    plab.savefig(fn)
    plab.close()
    return fn


def plot_alignment_to_numpy(alignment, info=None):
    # pdb.set_trace()
    fig, ax = plab.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plab.xlabel(xlabel)
    plab.ylabel('Encoder timestep')
    plab.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plab.close()
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plab.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plab.colorbar(im, ax=ax)
    plab.xlabel("Frames")
    plab.ylabel("Channels")
    plab.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plab.close()
    return data


def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plab.subplots(figsize=(12, 3))
    ax.scatter(list(range(len(gate_targets))), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(list(range(len(gate_outputs))), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plab.xlabel("Frames (Green target, Red predicted)")
    plab.ylabel("Gate State")
    plab.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plab.close()
    return data

