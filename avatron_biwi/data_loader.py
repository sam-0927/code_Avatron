import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
import librosa    

from text import *
from text import _arpabet_to_sequence
from text.cleaners import english_cleaners
import pdb

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]    
        speaker = self.data[index]["spk"]
        audio = self.data[index]["audio"]
        text = self.data[index]["text"]
        pho_label = self.data[index]["pho_label"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        mel = self.data[index]["mel"]
        f0 = self.data[index]["f0"]
        energy = self.data[index]["energy"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        return torch.FloatTensor(mel), torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), \
        torch.LongTensor(text), torch.LongTensor(pho_label), torch.FloatTensor(f0), torch.FloatTensor(energy), file_name, speaker

    def __len__(self):
        return self.len

def get_mel(x):
    x = torch.FloatTensor(x.astype(np.float32))
    mel = stft.mel_spectrogram(x.unsqueeze(0))
    return mel.squeeze(0)

def get_text(wav_name):
    text_name = wav_name.split('_')[-1][:-4] #sentece01
    text_idx = int(text_name[-2:])-1
    idx_and_text = texts[text_idx]
    idx, text = idx_and_text.split('|')
    text = np.array(text_to_sequence(text, ['english_cleaners']))

    return text

unround_label = ['eh0','eh1','ih0','ih1','iy0','iy1']
round_label = ['er0','er1','ow0','ow1','oy0','oy1','uw0','uw1','uh0','uh1']
both_label = ['aa','aa0','aa1','aa2','aw','aw0','aw1','aw2','ay','ay0','ay1','ay2']
bilabial_cons_label = ['b','m','p'] # m,b,p
unround_idx, round_idx, bilabial_cons_idx, both_idx = [], [], [], []
for i in range(len(unround_label)):
    unround_idx.append(_arpabet_to_sequence(unround_label[i].upper()))
for i in range(len(round_label)):
    round_idx.append(_arpabet_to_sequence(round_label[i].upper()))
for i in range(len(bilabial_cons_label)):
    bilabial_cons_idx.append(_arpabet_to_sequence(bilabial_cons_label[i].upper()))
for i in range(len(both_label)):
    both_idx.append(_arpabet_to_sequence(both_label[i].upper()))
round_idx = np.array(round_idx)
unround_idx = np.array(unround_idx)
bilabial_cons_idx = np.array(bilabial_cons_idx)
both_idx = np.array(both_idx)
round_idx = round_idx.reshape(round_idx.shape[0])
unround_idx = unround_idx.reshape(unround_idx.shape[0])
bilabial_cons_idx = bilabial_cons_idx.reshape(bilabial_cons_idx.shape[0])
both_idx = both_idx.reshape(both_idx.shape[0])

def read_data(args, hp):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    valid_test_data = []
    test_data = []
    base_path = 'dataset_root_path'
    vertices_path = os.path.join(base_path, args.vertices_path)
    filepath = args.all_file

    template_file = os.path.join(base_path, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    with open(filepath, 'r') as fd:
        fs = fd.readlines()
        for f in tqdm(fs):
            wav_path, text, spk_id = f.strip().split('|')
            base_path_list = wav_path.split('/')[:-2]
            data_base_path = '/'.join(base_path_list)
            name = wav_path.strip().split('/')[-1]
            speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
            input_values = torch.FloatTensor(speech_array.astype(np.float32))
            mel_path = os.path.join(data_base_path, 'mel', '{}-mel-{}.npy'.format(hp.dataset, name[:-4]))
            mel_values = torch.from_numpy(np.load(mel_path))
            f0_path = os.path.join(data_base_path, 'f0', '{}-f0-{}.npy'.format(hp.dataset, name[:-4]))
            f0_values = np.load(f0_path)
            energy_path = os.path.join(data_base_path, 'energy', '{}-energy-{}.npy'.format(hp.dataset, name[:-4]))
            energy_values = np.load(energy_path)
            text = np.array(text_to_sequence(text, []))
            pho_label = []
            for i in range(len(text)):
                if text[i] in round_idx:
                    pho_label.append(1)
                elif text[i] in unround_idx:
                    pho_label.append(2)
                elif text[i] in bilabial_cons_idx:
                    pho_label.append(3)
                elif text[i] in both_idx:
                    pho_label.append(4)
                else:
                    pho_label.append(0)
            pho_label = np.array(pho_label)
            text_values = text
            f = name[:-4] + '.wav'
            key = f.replace('wav', 'npy')
            data[key]["audio"] = input_values
            data[key]["mel"] = mel_values
            data[key]["f0"] = f0_values / hp.f0_max
            data[key]["energy"] = energy_values / hp.energy_max
            data[key]["text"] = text_values
            data[key]["pho_label"] = pho_label
            subject_id = name.split('_')[0]
            temp = templates[subject_id]
            data[key]["name"] = f
            data[key]["spk"] = spk_id
            data[key]["template"] = temp.reshape((-1)) 
            vertice_path = os.path.join(vertices_path, f.replace('wav','npy'))
            if not os.path.exists(vertice_path):
                del data[key]
            else:
                if args.dataset == "vocaset":
                    data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]
                elif args.dataset == "BIWI":
                    data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]
    
    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = k.split('_')[0]
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['test']:
            valid_test_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)
    return train_data, valid_data, valid_test_data, test_data, subjects_dict

def get_dataloaders(args, hp):
    dataset = {}
    train_data, valid_data, valid_test_data, test_data, subjects_dict = read_data(args, hp)
    train_data = Dataset(train_data,subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data,subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    valid_test_data = Dataset(valid_test_data,subjects_dict,"test")
    dataset["valid_test"] = data.DataLoader(dataset=valid_test_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data,subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset

if __name__ == "__main__":
    get_dataloaders()
    
