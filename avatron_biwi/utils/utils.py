import os
from torch.utils.data import DataLoader
import torch
from text import *
import matplotlib.pyplot as plt
import random
import pdb
def plot_attn(train_logger, enc_attns, pros_attn, soft_A, hard_A, current_step):
    # pdb.set_trace()
    idx = random.randint(0, enc_attns[0].size(0) - 1)
    for i in range(len(enc_attns)):
        train_logger.add_figure(
            "encoder.attns_layer_%s"%i,
            plot_alignment(enc_attns[i].data.cpu().numpy()[idx].T),
            current_step)

    idx2 = random.randint(0, pros_attn.size(0) - 1)
    train_logger.add_figure(
            "encoder.prosody",
            plot_alignment(pros_attn.data.cpu().numpy()[idx2].T),
            current_step)
    
    idx1 = random.randint(0, soft_A.size(0) - 1)
    train_logger.add_figure(
            "soft alignment",
            plot_alignment(soft_A.data.cpu().numpy()[idx1]),
            current_step)

    train_logger.add_figure(
            "hard alignment",
            plot_alignment(hard_A.data.cpu().numpy()[idx1]),
            current_step)

def v_plot_attn(train_logger,pros_attn, soft_A, hard_A, current_step, j):
    train_logger.add_figure(
            "validation.prosody_{}".format(j),
            plot_alignment(pros_attn.data.cpu().numpy()[0].T),
            current_step)
    
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


def load_checkpoint(checkpoint_path, model, optimizer=None):
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')   # pretrained
    model_dict = model.state_dict()
    if 'tts' in checkpoint_path:
        pretrained_dict = {k: v for k,v in checkpoint_dict['model'].items() if k in model_dict.keys()}
    else:
        pretrained_dict = {k: v for k,v in checkpoint_dict['state_dict'].items() if k in model_dict.keys()}
    #pretrained_dict = {k: v for k,v in checkpoint_dict['state_dict'].items() if k in model_dict.keys()}

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    iteration = None
    if 'tts' not in checkpoint_path:
        iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    if optimizer is None:
        return model
    if 'tts' in checkpoint_path:
        pretrained_opt = {k: v for k,v in checkpoint_dict['optim_g'].items() if k in model_dict.keys()}
    else:
        pretrained_opt = {k: v for k,v in checkpoint_dict['optimizer'].items() if k in model_dict.keys()}
    #pretrained_opt = {k: v for k,v in checkpoint_dict['optimizer'].items() if k in model_dict.keys()}

    model_dict.update(pretrained_opt)
    model.load_state_dict(model_dict)
    #optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, f'{filepath}/checkpoint_{iteration}')

def tts_save_checkpoint(model, optimizer, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, f'{filepath}/tts_checkpoint_{iteration}')
def D_save_checkpoint(model, optimizer, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, f'{filepath}/D_checkpoint_{iteration}')

'''
def lr_scheduling(opt, step, init_lr=hparams.lr, warmup_steps=hparams.warmup_steps):
    opt.param_groups[0]['lr'] = init_lr * min(step ** -0.5, step * warmup_steps ** -1.5)
    return
'''

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len))
    mask = (lengths.unsqueeze(1) <= ids.cuda()).to(torch.bool)
    return mask


def get_mask(lengths):
    mask = torch.zeros(len(lengths), torch.max(lengths)).cuda()
    for i in range(len(mask)):
        mask[i] = torch.nn.functional.pad(torch.arange(1,lengths[i]+1),[0,torch.max(lengths)-lengths[i]],'constant')
    return mask.cuda()

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

