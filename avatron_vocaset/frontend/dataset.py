import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

from utils_pas import pad_1D, pad_2D, process_meta
from text import text_to_sequence, sequence_to_text
import pdb
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Dataset(Dataset):
    def __init__(self, filename="filelists/libritts_vctk/train.txt", sort=True):
        self.path, self.text, self.spk = process_meta(filename)
        self.sort = sort

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.path[idx].split('/')[-1][:-4]
        speaker = self.spk[idx]
        if 'libritts' in self.path[idx]:
            dataset = 'libritts'
        elif 'vctk' in self.path[idx]:
            dataset = 'vctk'
        elif 'vocaset' in self.path[idx]:
            dataset = 'vocaset'
        elif 'BIWI' in self.path[idx]:
            dataset = 'biwi'
        else:  
            raise Exception('Unknow dataset!')
        
        phone = np.array(text_to_sequence(self.text[idx], []))
        path_list = self.path[idx].split('/')
        mel_path = os.path.join('/'.join(path_list[:-1]), "mel", "{}-mel-{}.npy".format(dataset, basename))
        mel_target = np.load(mel_path)
        f0_path = os.path.join('/'.join(path_list[:-1]), "f0", "{}-f0-{}.npy".format(dataset, basename))
        f0 = np.load(f0_path)
        energy_path = os.path.join('/'.join(path_list[:-1]), "energy", "{}-energy-{}.npy".format(dataset, basename))
        energy = np.load(energy_path)
        
        sample = {"id": basename,
                  "speaker": speaker,
                  "text": phone,
                  "mel_target": mel_target,
                  "f0": f0,
                  "energy": energy,
                  "dataset":dataset}

        return sample

    def reprocess(self, batch):
        ids, spks, texts, mel_targets, f0s, energies, dataset = [], [], [], [], [], [], []
        for d in batch:
            #d[1]
            ids.append(d['id'])
            spks.append(d['speaker'])
            texts.append(d['text'])
            mel_targets.append(d['mel_target'])
            f0s.append(d['f0'])
            energies.append(d['energy'])
            dataset.append(d['dataset'])

        length_text = np.array(list())
        for text in texts:
            length_text = np.append(length_text, text.shape[0])

        length_mel = np.array(list())
        for mel in mel_targets:
            length_mel = np.append(length_mel, mel.shape[0])
        
        texts = pad_1D(texts)
        mel_targets = pad_2D(mel_targets)
        f0s = pad_1D(f0s)
        energies = pad_1D(energies)

        out = {"id": ids,
               "speaker": spks,
               "text": texts,
               "mel_target": mel_targets,
               "f0": f0s,
               "energy": energies,
               "src_len": length_text,
               "mel_len": length_mel,
               "dataset":dataset}
        
        return out

    def collate_fn(self, batch):
        output = self.reprocess(batch)
        return output
