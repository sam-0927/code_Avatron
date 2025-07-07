import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from sklearn.utils import shuffle
import pdb
import sys
def intercross(mels, names, emos, spks):
    # Search intercross pair
    new_mels = torch.zeros_like(mels).cuda()
    new_names = []
    new_emos = []
    for i in range(len(mels)):
        source_emo, source_spk = emos[i], spks[i]    
        replace = False
        for j in range(len(mels)):
            emo, spk = emos[j], spks[j]
            if emo == source_emo and spk != source_spk:
                replace = True
                new_names.append(names[j])
                new_emos.append(emos[j])
                new_mels[i] = mels[j]
                break
        if replace == False:
            new_names.append(names[i])
            new_emos.append(emos[i])
            new_mels[i] = mels[i]
    #pdb.set_trace()
    return new_mels

class Style_Encoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.encoder = ReferenceEncoder(hparams)

    def forward(self, inputs):
        style_emb = self.encoder(inputs)  # [B, 256] 
        return style_emb

class ReferenceEncoder(nn.Module):
    def __init__(self,hparams):
        super().__init__()
        self.ref_enc_filters = hparams.ref_enc_filters
        self.mel_dim = hparams.mel_dim
        self.E = hparams.E
        K = len(self.ref_enc_filters)
        filters = [1] + self.ref_enc_filters
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=self.ref_enc_filters[i]) for i in range(K)])

        out_channels = self.calculate_channels(self.mel_dim, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=self.ref_enc_filters[-1] * out_channels,
                          hidden_size=self.E // 2,
                          batch_first=True,
                          bidirectional=True)
        self.tanh = nn.Tanh()
    
    def forward(self, inputs):
        N = inputs.size(0)
        out = inputs.view(N, 1, -1, self.mel_dim)  # [N, 1, Ty, n_mels]
        
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, E]
        out = torch.cat((out[0,:,:], out[1,:,:]), dim=-1)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class Speaker_Encoder(nn.Module):
    def __init__(self, hp):
        super(Speaker_Encoder, self).__init__()
        self.embedding_libri = nn.Embedding(hp.num_speaker_libri+1, hp.spk_hidden_dim)
        self.speaker_embedding_libri = nn.Sequential(nn.Linear(hp.spk_hidden_dim, hp.encoder_hidden),
                                               nn.LeakyReLU(),
                                               nn.Linear(hp.encoder_hidden, hp.encoder_hidden))
        self.embedding_esd = nn.Embedding(hp.num_speaker_esd+1, hp.spk_hidden_dim)
        self.speaker_embedding_esd = nn.Sequential(nn.Linear(hp.spk_hidden_dim, hp.encoder_hidden),
                                               nn.LeakyReLU(),
                                               nn.Linear(hp.encoder_hidden, hp.encoder_hidden))
        self.embedding_biwi = nn.Embedding(hp.num_speaker_biwi+1, hp.spk_hidden_dim)
        self.speaker_embedding_biwi = nn.Sequential(nn.Linear(hp.spk_hidden_dim, hp.encoder_hidden),
                                               nn.LeakyReLU(),
                                               nn.Linear(hp.encoder_hidden, hp.encoder_hidden))

    def speaker_emb(self, spk_id, dataset):
        #pdb.set_trace()
#print(spk_id, dataset)
        spk_id = torch.from_numpy(np.array(spk_id)).long().cuda()
        for i in range(len(dataset)):
            if dataset[i] == 'libritts':
                embedding = self.embedding_libri(spk_id[i])
                spk_emb_ = self.speaker_embedding_libri(embedding)
            elif dataset[i] == 'esd':
                embedding = self.embedding_esd(spk_id[i])
                spk_emb_ = self.speaker_embedding_esd(embedding)
            elif dataset[i] == 'biwi':
                embedding = self.embedding_biwi(spk_id[i])
                spk_emb_ = self.speaker_embedding_biwi(embedding)
            spk_emb_ = spk_emb_.unsqueeze(0)
            if i == 0:
                spk_emb = spk_emb_
            else:
                spk_emb = torch.cat((spk_emb, spk_emb_), dim=0)
        return spk_emb




class Emotion_Encoder(nn.Module):
    def __init__(self, hp):
        super(Emotion_Encoder, self).__init__()
        self.embedding = nn.Embedding(hp.num_emotion+1, hp.spk_hidden_dim)
        self.emotion_embedding = nn.Sequential(nn.Linear(hp.spk_hidden_dim, hp.encoder_hidden),
                                               nn.LeakyReLU(),
                                               nn.Linear(hp.encoder_hidden, hp.encoder_hidden))

    def emotion_emb(self, emo):
        #pdb.set_trace()
        emo_id = torch.from_numpy(np.array(emo)).long().cuda()
        embedding = self.embedding(emo_id)
        emo_emb = self.emotion_embedding(embedding)
        return emo_emb


