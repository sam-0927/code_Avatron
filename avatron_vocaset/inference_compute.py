import re, random, math
import numpy as np
import argparse
from tqdm import tqdm
import os, shutil
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import get_dataloaders
from avatron import Avatron
import pdb
from utils.writer import get_writer
from utils.utils import *

from frontend.models import FrontEnd 
from frontend.loss import Loss, guide_loss
from align_loss import AlignLoss
import hparams as hp

from hifigan import Generator
import json
from env import AttrDict
from scipy.io.wavfile import write
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from g2p_en import G2p
MAX_WAV_VALUE = 32768
g2p = G2p()
idx = np.load('lip_idx_342.npy')
mouth_map = [i for i in idx]
def preprocess(text):
    phone = g2p(text)
    phone = '{'+ '}{'.join(phone) + '}'
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    phone = phone.replace('}{', ' ')
    sequence = np.array(text_to_sequence(phone, []))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)

def synthesizer(args, dev_loader, model, tts_model):
    result_path = os.path.join(args.dataset, args.result_path)
    os.makedirs(result_path, exist_ok=True)
    # Vocoder
    config_file = 'config.json'
    with open(config_file) as f:
        data = f.read()
    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    vocoder = Generator(h).to(device)
    state_dict_g = torch.load('pretrained_hifigan_checkpoint_path', map_location=device)        
    vocoder.load_state_dict(state_dict_g['generator'])
    vocoder.eval()
    vocoder.remove_weight_norm()

    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.dataset, args.save_path, 'checkpoint_'+args.checkpoint)
        model = load_checkpoint(checkpoint_path, model)
    tts_checkpoint_path = os.path.join('pretrained_TTS_checkpoint_path')
    tts_model = load_checkpoint(tts_checkpoint_path, tts_model)

            
    train_subjects_list = [i for i in args.train_subjects.split(" ")]    
    model.eval()
    tts_model.eval()
    losses, lip_losses, lip_aver_losses = [], [], []
    with torch.no_grad():
        for mel, vertice, template, one_hot_all, text, _, pho_label, f0, energy, file_name, speaker in dev_loader:
            # to gpu
            mel, vertice, template, one_hot_all = \
            mel.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot_all.to(device="cuda") 
            text, f0, energy = \
            text.to(device="cuda"), f0.to(device="cuda"), energy.to(device="cuda")
            pho_label = pho_label.to(device="cuda")
                    
            src_len = torch.LongTensor([text.size(1)]).to(device="cuda")
            mel_len = torch.LongTensor([mel.size(1)]).to(device="cuda")

            style_path = 'style_dir_vocaset/'+speaker[0]+'_i2i.npy'
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
            # Synthesize speech
            if len(mel_output_pro.size()) != 3:
                mel_output_pro = mel_output_pro.unsqueeze(0)
            mel_output_pro = mel_output_pro.transpose(1,2)

            y_g_hat = vocoder(mel_output_pro)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.detach().cpu().numpy().astype('int16')
            
            output = os.path.join(result_path, file_name[0].split('.')[0]+'.wav')
            write(output, hp.sampling_rate, audio)
            train_subject = "_".join(file_name[0].split("_")[:-1])
            condition_subject = train_subject
            
            if condition_subject in train_subjects_list:
                iter = train_subjects_list.index(condition_subject)
                one_hot = one_hot_all[:,iter,:]
                prediction = model.predict(up_emb, up_pho_emb, template, one_hot, None, args.up_pho, vertice.size(1))

                loss = ((prediction - vertice)**2)

                # Compute average lip loss
                ref = vertice.squeeze().detach().cpu().numpy()
                ref = ref.reshape(-1, args.vertice_dim//3, 3)
                pred = prediction.squeeze().detach().cpu().numpy()
                pred = pred.reshape(-1, args.vertice_dim//3, 3)
                l2_dist_mouth = np.array([np.square(ref[:,v, :]-pred[:,v,:]) for v in mouth_map])
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

                # Compute vertice loss
                losses.append(torch.mean(loss).item())

                np.save(os.path.join(result_path,file_name[0].split('.')[0]+'.npy'), \
                prediction.squeeze(0).detach().cpu().numpy())
                print(file_name[0].split('.')[0]+'.npy')
                print(torch.mean(loss), l2_dist_max_mouth, np.mean(lip_loss),'\n')
            else:
                print('unseen speaker!')
        print('Total:{} Lip:{} Lip average:{}'.format(np.mean(losses), np.mean(lip_losses), np.mean(lip_aver_losses)))
                
            
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='Avatron')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    # TTS
    parser.add_argument("--adam_b1", default=0.5)
    parser.add_argument("--adam_b2", default=0.9)
    # Avatar
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=256, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--wav_path", type=str, default= "trim", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy_trim", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=200, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--template_file", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--save_path", type=str, default="save_ln_40", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA"
       " FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
       " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA"
       " FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA"
       " FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--val_subjects", type=str, default="FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    

    parser.add_argument("--n_symbols", type=int, default=153)
    parser.add_argument("--symbols_embedding_dim", type=int, default=256)
    parser.add_argument("-c", "--checkpoint", type=str, default='187')
    parser.add_argument("--up_pho", default=True)

    parser.add_argument("--all_file", type=str, default='filelists/vocaset/all.txt')

    args = parser.parse_args()

    #build model
    model = Avatron(args)
    tts_model = FrontEnd()
    print("Avatar model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    model = model.to(torch.device("cuda"))
    tts_model = tts_model.to(torch.device("cuda"))

    #load data
    dataset = get_dataloaders(args, hp)

    synthesizer(args, dataset["valid_test"], model, tts_model)
    
if __name__=="__main__":
    main()
