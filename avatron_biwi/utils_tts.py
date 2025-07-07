import pdb
import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt
import re
import numpy as np
import random
from frontend.dataset import Dataset

def get_dataset_filelist(a, hp):
    training_data = Dataset(a.input_training_file, hp)
    validation_data = Dataset(a.input_validation_file, hp) 
    return training_data, validation_data

def plot_attn(train_logger, enc_attns, soft_A, hard_A, current_step, hp):
    # pdb.set_trace()
    idx = random.randint(0, enc_attns[0].size(0) - 1)
    for i in range(len(enc_attns)):
        train_logger.add_figure(
            "encoder.attns_layer_%s"%i,
            plot_alignment(enc_attns[i].data.cpu().numpy()[idx].T),
            current_step)
    '''
    idx2 = random.randint(0, pros_attn.size(0) - 1)
    train_logger.add_figure(
            "encoder.prosody",
            plot_alignment(pros_attn.data.cpu().numpy()[idx2].T),
            current_step)
    '''
    idx1 = random.randint(0, soft_A.size(0) - 1)
    train_logger.add_figure(
            "soft alignment",
            plot_alignment(soft_A.data.cpu().numpy()[idx1]),
            current_step)

    train_logger.add_figure(
            "hard alignment",
            plot_alignment(hard_A.data.cpu().numpy()[idx1]),
            current_step)

def v_plot_attn(train_logger, soft_A, hard_A, current_step, hp, j):
    '''
    train_logger.add_figure(
            "validation.prosody_{}".format(j),
            plot_alignment(pros_attn.data.cpu().numpy()[0].T),
            current_step)
    '''
    train_logger.add_figure(
            "validation.soft alignment_{}".format(j),
            plot_alignment(soft_A.data.cpu().numpy()[0]),
            current_step)

    train_logger.add_figure(
            "validation.hard alignment_{}".format(j),
            plot_alignment(hard_A.data.cpu().numpy()[0]),
            current_step)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
   

def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def plot_alignment(alignment):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def last_num_return(cp_dir):
    return int(cp_dir.split('/')[-1].split('.')[0].split('_')[-1])

def scan_checkpoint_tts(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*.tar')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return sorted(cp_list)[-1], -1
    else:
        num = max([int(re.findall("\d+", cp_one)[-1]) for cp_one in cp_list])
        for cp_one in cp_list:
            if str(num) in cp_one: 
                return cp_one, num

# from dathudeptrai's FastSpeech2 implementation
def standard_norm(x, mean, std, is_mel=False):

    if not is_mel:
        x = remove_outlier(x)

    zero_idxs = np.where(x == 0.0)[0]
    x = (x - mean) / std
    x[zero_idxs] = 0.0
    return x

    return (x - mean) / std

def de_norm(x, mean, std):
    zero_idxs = torch.where(x == 0.0)[0]
    x = mean + std * x
    x[zero_idxs] = 0.0
    return x


def _is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)

    return np.logical_or(x <= lower, x >= upper)


def remove_outlier(x):
    """Remove outlier from x."""
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)

    indices_of_outliers = []
    for ind, value in enumerate(x):
        if _is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)

    x[indices_of_outliers] = 0.0
    # replace by mean f0.
    x[indices_of_outliers] = np.max(x)
    return x

def average_by_duration(x, durs):
    mel_len = sum(durs)#durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    #x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    x_char = np.zeros((np.asarray(durs).shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)
