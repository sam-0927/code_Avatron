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

random_seed =1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)


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
    losses, lip_losses, fdd_losses = [], [], []
    lip_aver_losses = []
    with torch.no_grad(): 
        for mel, vertice, template, one_hot_all, text, pho_label, f0, energy, file_name, speaker in dev_loader:
            if file_name[0].split('.')[0] == 'F2_40':
                print('trimming error')
                continue
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
            
            ###### TTS forward
            up_emb, mel_predicted, log_duration_output, src_mask, mel_mask, _,\
            enc_attns, \
            soft_A, hard_A, duration_target, perf_mask,\
            v_mel_output, post_p, post_e, mel_output_pro, target_p_pho, target_e_pho \
            = tts_model.validation(style_emb, None, 100000+10, text, src_len, f0, energy,\
                mel, mel_len)

            pho_emb = model.pho_emb(pho_label)
            pho_emb = model.pho_fc(pho_emb)
            up_pho_emb = torch.matmul(hard_A.transpose(1,2), pho_emb)   
           
            # Vocoder
            gen_speech = hifigan(mel_predicted.transpose(1,2))
            audio = gen_speech.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.detach().cpu().numpy().astype('int16')
            output = os.path.join(result_path, file_name[0].split('.')[0]+'.wav')
            write(output, hp.sampling_rate, audio)
            # Avatar
            train_subject = speaker[0]
            condition_subject = train_subject
            if condition_subject in train_subjects_list:
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]

                prediction, vertice_ = model.predict(up_emb, up_pho_emb, template, one_hot, None, args.up_pho, vertice, vertice.size(1))

                loss = ((prediction - vertice_)**2)

                # Compute lip loss
                ref = vertice_.squeeze().detach().cpu().numpy()
                ref = ref.reshape(-1, args.vertice_dim//3, 3)
                pred = prediction.squeeze().detach().cpu().numpy()
                pred = pred.reshape(-1, args.vertice_dim//3, 3)
                l2_dist_mouth = np.array([np.square(ref[:,v, :]-pred[:,v,:]) for v in mouth_map])   # [4996, frame, 3]
                l2_dist_mouth = np.transpose(l2_dist_mouth, (1,0,2))    # [frame, 4996, 3]
                l2_dist_mouth = np.sum(l2_dist_mouth, axis=2)
                l2_dist_max_mouth = np.max(l2_dist_mouth, axis=1)           # frame select
                l2_dist_max_mouth = np.mean(l2_dist_max_mouth)
                lip_losses.append(l2_dist_max_mouth)

                # Compute average lip loss
                ref_mouth = np.array([ref[:,v,:] for v in mouth_map])
                pred_mouth = np.array([pred[:,v,:] for v in mouth_map])
                lip_loss = ((pred_mouth - ref_mouth)**2)
                lip_aver_losses.append(np.mean(lip_loss))

                # Compute FDD loss
                ref = vertice_ - template.unsqueeze(1)
                ref = ref.squeeze().detach().cpu().numpy()
                ref = ref.reshape(-1, args.vertice_dim//3, 3)
                ref_l2_dist_upper = np.array([np.square(ref[:,v, :]) for v in upper_map])
                ref_l2_dist_upper = np.transpose(ref_l2_dist_upper, (1,0,2))
                ref_l2_dist_upper = np.sum(ref_l2_dist_upper, axis=2)
                ref_l2_dist_upper = np.std(ref_l2_dist_upper, axis=0)
                ref_upper_std = np.mean(ref_l2_dist_upper)

                pred = prediction - template.unsqueeze(1)
                pred = pred.squeeze().detach().cpu().numpy()
                pred = pred.reshape(-1, args.vertice_dim//3, 3)
                pred_l2_dist_upper = np.array([np.square(pred[:,v, :]) for v in upper_map])
                pred_l2_dist_upper = np.transpose(pred_l2_dist_upper, (1,0,2))
                pred_l2_dist_upper = np.sum(pred_l2_dist_upper, axis=2)
                pred_l2_dist_upper = np.std(pred_l2_dist_upper, axis=0)
                pred_upper_std = np.mean(pred_l2_dist_upper)
                fdd_losses.append(ref_upper_std - pred_upper_std)

                # Compute vertice loss
                losses.append(torch.mean(loss).item())

                print(file_name[0].split('.')[0]+'.npy')
            else:
                print('unseen speaker!')
                
        print('Total:{} Lip:{} Lip average:{} FDD:{}'.format(np.mean(losses), np.mean(lip_losses), np.mean(lip_aver_losses), np.mean(fdd_losses)))

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

    parser.add_argument("--checkpoint", '-c', type=str, default="25")
    parser.add_argument("--all_file", type=str, default='filelists/biwi/all.txt')
    parser.add_argument('--up_pho', required=True)

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

    synthesizer(args, dataset["valid_test"], model, tts_model)
    
if __name__=="__main__":
    main()
