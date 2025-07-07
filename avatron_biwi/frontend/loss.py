import torch
import torch.nn as nn
import hparams as hp
import torch.nn.functional as F
import pdb

eps = 1e-8

def ClassificationLoss(logit, label):
    loss = torch.sum(logit*label, dim=-1) 
    loss = torch.mean(-torch.log(loss + eps))
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

def make_onehot(idx):
    onehot = torch.zeros(4)
    onehot[idx] = 1
    return onehot

def make_code(label):
    idx = None;
    for i, speaker in enumerate(['m1', 'f1', 'm2', 'f2']):
        if label == speaker:
            idx = i
    return make_onehot(idx)


def guide_loss(alignments, text_lengths, mel_lengths):
    alignments = alignments.transpose(1,2)
    B, T, L = alignments.size()

    # B, T, L
    W = alignments.new_zeros(B, T, L)
    mask = alignments.new_zeros(B, T, L)
    
    for i, (t, l) in enumerate(zip(mel_lengths, text_lengths)):
        mel_seq = alignments.new_tensor( torch.arange(t).to(torch.float32).unsqueeze(-1).cuda()/t)
        text_seq = alignments.new_tensor( torch.arange(l).to(torch.float32).unsqueeze(0).cuda()/l)
        x = torch.pow(mel_seq-text_seq, 2)
        W[i, :t, :l] += alignments.new_tensor(1-torch.exp(-3.125*x))
        mask[i, :t, :l] = 1
    losses = alignments*W
    
    return torch.mean(losses.masked_select(mask.unsqueeze(1).unsqueeze(1).to(torch.bool)))


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, log_d_predicted, log_d_target,  p_target, p_post,
                e_target, e_post, mel_predicted, mel_target, 
                src_mask, mel_mask, step, p_target_pho, e_target_pho, mel_pred_pro):

        log_d_target.requires_grad = False
        p_target.requires_grad = False
        e_target.requires_grad = False
        mel_target.requires_grad = False # 
        
        d_pred = torch.exp(log_d_predicted) - 1
        d_tar = torch.exp(log_d_target) - 1
        sum_pred = torch.sum(d_pred, 1)
        sum_target = torch.sum(d_tar, 1)
        sum_loss = self.mae_loss(sum_target, sum_pred)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_d_predicted = log_d_predicted.masked_select(src_mask)
        log_d_target = log_d_target.masked_select(src_mask)
        mel_predicted = mel_predicted.masked_select(mel_mask.unsqueeze(-1)) # actually mfcc
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(-1)) #  

        d_loss = self.mae_loss(log_d_predicted, log_d_target)
        
        src_mask = ~src_mask
        src_mask = src_mask.to(device)
        m_loss = self.mse_loss(mel_predicted, mel_target)       ## L2
        m_pro_loss = None
        if mel_pred_pro is not None:
            mel_pred_pro = mel_pred_pro.masked_select(mel_mask.unsqueeze(-1))
            m_pro_loss = self.mse_loss(mel_pred_pro, mel_target)

        post_p_loss = self.mae_loss(p_post, p_target_pho)
        post_e_loss = self.mae_loss(e_post, e_target_pho)

        return d_loss, m_loss, sum_loss, post_p_loss, post_e_loss, m_pro_loss



