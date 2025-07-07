import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptflops

from data_loader import get_dataloaders
from avatron import Avatron
import pdb
from utils.utils import *

from frontend.models import FrontEnd 
import hparams as hp

from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

import json
from env import AttrDict
from hifigan import Generator
MAX_WAV_VALUE = 32768
from scipy.io.wavfile import write
with open('regions/lve.txt', 'r') as f:
    maps = f.read().split(", ")
    mouth_map = [int(i) for i in maps]
with open('regions/fdd.txt', 'r') as f:
    maps = f.read().split(", ")
    upper_map = [int(i) for i in maps]


def synthesizer(args, dev_loader, model, tts_model):
    result_path = os.path.join(args.dataset,args.result_path)
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(args.dataset, args.save_path)
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(save_path, 'checkpoint_'+args.checkpoint)
        model = load_checkpoint(checkpoint_path, model)
    tts_checkpoint_path = os.path.join('pretrained_TTS_checkpoint_path')
    tts_model = load_checkpoint(tts_checkpoint_path, tts_model)
    
    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    
    # Vocoder
    config_file = 'config.json'
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    
    hifigan = Generator(h).to(device="cuda")
    vocoder_dict = torch.load('pretrained_hifigan_checkpoint_path')
    hifigan.load_state_dict(vocoder_dict['generator'])
    hifigan.eval()
    hifigan.remove_weight_norm()
    
    model.eval()
    tts_model.eval()
    lip_var_list = []
    with torch.no_grad(): 
        for mel, vertice, template, one_hot_all, text, pho_label, f0, energy, file_name, speaker in dev_loader:
            # to gpu
            mel, vertice, template, one_hot_all = \
            mel.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda") 
            text, f0, energy = \
            text.to(device="cuda"), f0.to(device="cuda"), energy.to(device="cuda")
            pho_label = pho_label.to(device="cuda")
           
            src_len = torch.LongTensor([text.size(1)]).to(device="cuda")
            mel_len = torch.LongTensor([mel.size(1)]).to(device="cuda")
            style_path = 'style_dir_biwi/'+speaker[0]+'_i2i.npy'
            style_emb = torch.from_numpy(np.load(style_path)).to(device="cuda")
            
            ### ## TTS forward
            up_emb, mel_predicted, d_prediction = tts_model.inference(None, text, src_len, None, None, style_emb)         
           
            # Vocoder
            # reference
            gen_speech = hifigan(mel.transpose(1,2))
            audio = gen_speech.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.detach().cpu().numpy().astype('int16')
            output = os.path.join(result_path, 'ref_'+file_name[0].split('.')[0]+'.wav')
            write(output, hp.sampling_rate, audio)

            gen_speech = hifigan(mel_predicted.transpose(1,2))
            audio = gen_speech.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.detach().cpu().numpy().astype('int16')
            output = os.path.join(result_path, file_name[0].split('.')[0]+'.wav')
            write(output, hp.sampling_rate, audio)
            pho_emb = model.pho_emb(pho_label)
            pho_emb = model.pho_fc(pho_emb)
            up_pho_emb,_,_ = tts_model.variance_adaptor.inference(pho_emb, d_prediction)
            
            # Avatar
            train_subject = speaker[0]
            condition_subject = train_subject
            lip_var_spk_list = []
            if condition_subject in train_subjects_list:
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                prediction, vertice_ = model.predict(up_emb, template, one_hot, vertice, vertice.size(1))
                np.save(os.path.join(result_path,file_name[0].split('.')[0]+'.npy'), prediction.squeeze(0).detach().cpu().numpy())
                print(file_name[0].split('.')[0]+'.npy')
            else:
                for iter in range(one_hot_all.shape[-1]):
                    condition_subject = train_subjects_list[iter]
                    one_hot = one_hot_all[:,iter,:]
                    prediction, _ = model.predict(up_emb, up_pho_emb, template, one_hot, audio.shape[0], args.up_pho, None, None)
                    pred = prediction.squeeze().detach().cpu().numpy()
                    pred = pred.reshape(-1, args.vertice_dim//3, 3)
                    lip = np.array([pred[:,v, :] for v in mouth_map])
                    lip = np.transpose(lip, (1,0,2))
                    lip_var = np.mean(np.std(lip, axis=0))
                    lip_var_spk_list.append(lip_var)
        lip_var_list.append(lip_var_spk_list)
    print('average Var lip:{}'.format(np.std(np.mean(lip_var_list, axis=0))))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='Avatron')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    # Avatar
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=256, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "wavs_trim", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy_trim", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=800, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")

    parser.add_argument("--checkpoint", '-c', type=str, default=None)
    parser.add_argument("--all_file", type=str, default='filelists/biwi/all.txt')
    parser.add_argument("--up_pho", type=str, default=True)

    args = parser.parse_args()

    #build model
    model = Avatron(args)
    tts_model = FrontEnd()
    print("model parameters: ", count_parameters(model))
    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    tts_model = tts_model.to(torch.device("cuda"))

    #load data
    dataset = get_dataloaders(args, hp)

    synthesizer(args, dataset["test"], model, tts_model)
    
if __name__=="__main__":
    main()
