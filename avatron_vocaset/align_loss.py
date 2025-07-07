# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pdb

class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax=torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)
    
    def forward(self, attn_logprob, text_lens, mel_lens):
        '''
        Args:
        attn_logprob: (batch, 1, max_mel_len, max_src_len)
        text_lens: (b,)
        mel_lens: (b,)
        '''
        attn_logprob_pd = F.pad(input=attn_logprob,pad=(1,0,0,0,0,0,0,0),value=self.blank_logprob)
        cost_total = 0.0

        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq = target_seq.unsqueeze(0) #* (1, text_lens[bid])
            
            curr_logprob = attn_logprob_pd[bid].permute(1,0,2) #* (max_mel_len, 1, max_src_len)
            curr_logprob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,target_seq,input_lengths=mel_lens[bid:bid+1],target_lengths = text_lens[bid:bid+1])

            cost_total += cost
        
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total

class AlignLoss(torch.nn.Module):

    def __init__(self):
        super(AlignLoss, self).__init__()

        self.forwardsumloss = ForwardSumLoss()
        self.KLloss = nn.KLDivLoss()
    
    def forward(self, text_lens, mel_lens, soft_A, hard_A, perf_mask):
        log_soft_A = torch.log(soft_A+1e-7)
        log_soft_A = log_soft_A.masked_fill(perf_mask, 0)
        KLloss = self.KLloss(log_soft_A,hard_A)
        FSloss = self.forwardsumloss(log_soft_A.unsqueeze(1).transpose(2,3), text_lens, mel_lens)
        
        return FSloss, KLloss
