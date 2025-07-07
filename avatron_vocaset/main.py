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
from utils_tts import get_dataset_filelist, plot_spectrogram, plot_alignment, plot_attn, v_plot_attn

random_seed =1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
import time

def trainer(args, train_loader, dev_loader, model, tts_model, optimizer, steplr, criterion, epoch):
    save_path = os.path.join(args.dataset,args.save_path)
    os.makedirs(save_path, exist_ok=True)
    iteration = 0
    init_e = 0
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(save_path, 'checkpoint_'+args.checkpoint)
        model, optimizer, init_e = load_checkpoint(checkpoint_path, model, optimizer)
    tts_checkpoint_path = os.path.join('pretrained_TTS_checkpoint_path')
    tts_model = load_checkpoint(tts_checkpoint_path, tts_model)
   
    writer = get_writer(save_path)

    train_subjects_list = [i for i in args.train_subjects.split(" ")]
    
    min_loss = 10
    for e in range(init_e, epoch+1):
        loss_log = []
        loss_log_delta = []
        loss_log_delta2 = []
        tts_loss = Loss().to(device="cuda")
        align_criterion = AlignLoss().to(device="cuda")
        
        # train
        model.train()
        tts_model.eval()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        optimizer.zero_grad()
        times = []
        for i, (mel, vertice, template, one_hot, text, _, pho_label, f0, energy, file_name, speaker) in pbar:
            iteration += 1
            # to gpu
            mel, vertice, template, one_hot  = \
            mel.to(device="cuda"), vertice.to(device="cuda"), template.to(device="cuda"), one_hot.to(device="cuda")
            text, f0, energy =\
            text.to(device="cuda"), f0.to(device="cuda"), energy.to(device="cuda")
            pho_label = pho_label.to(device="cuda")

            # Predict acoustic features 
            src_len = torch.LongTensor([text.size(1)]).to(device="cuda")
            mel_len = torch.LongTensor([mel.size(1)]).to(device="cuda")
            style_path = 'style_dir_vocaset/'+speaker[0]+'_i2i.npy'
            style_emb = torch.from_numpy(np.load(style_path)).to(device="cuda")

            ###### TTS forward
            up_emb, mel_predicted, log_duration_output, src_mask, mel_mask, _,\
            enc_attns, \
            soft_A, hard_A, duration_target, perf_mask,\
            post_p, post_e, target_p_pho, target_e_pho \
            = tts_model(style_emb, None, text, src_len, f0, energy,\
	    		mel, mel_len)
            
            # Compute loss for acoustic model
            target_p = f0
            target_e = energy
            log_D = torch.log(duration_target + hp.log_offset)
            d_loss, m_loss, sum_loss, post_p_loss, post_e_loss, _ =\
            tts_loss(log_duration_output, log_D, target_p, post_p, 
                     target_e, post_e,
        	     mel_predicted, mel, ~src_mask, ~mel_mask, iteration, 
		     target_p_pho, target_e_pho, None)
             
            
            FSloss, KLloss = align_criterion(src_len, mel_len, soft_A, hard_A, perf_mask)
            g_loss = guide_loss(soft_A, src_len, mel_len)
            a_loss = FSloss+hp.kl_lamb*KLloss

            total_tts_loss = g_loss + d_loss + 5*m_loss + 10*a_loss + sum_loss + 5*post_p_loss + 5*post_e_loss
            
            start = time.time()
            # Avatar: Predict avatar given upsampled text embedding
            loss, pred_vert = \
            model(up_emb.detach(), pho_label, hard_A.detach(), template,  vertice, one_hot, criterion, args.up_pho)

            loss.backward()
            
            loss_log.append(loss.item())
            
            if i % args.gradient_accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()
            end = time.time()
            times.append(end-start)

            pbar.set_description("(Epoch {}, iteration {}) TRAIN LOSS:{:.9f} MEL:{:.2f} ALIGN:{:.2f} DP:{:.2f}".format((e+1), iteration ,np.mean(loss_log), m_loss.item(), a_loss.item(), d_loss.item()))
            
            writer.add_loss(np.mean(loss_log), iteration, 'Train/vertices')

            writer.add_loss(m_loss, iteration, 'Train/mel')
            writer.add_loss(a_loss, iteration, 'Train/align')
            writer.add_loss(d_loss, iteration, 'Train/duration')
            plot_attn(writer, enc_attns, soft_A, hard_A, iteration, hp)
        print('training time:{}'.format(np.sum(times)))
        # validation
        valid_loss_log = []
        valid_loss_log_delta = []
        valid_loss_log_delta2 = []
        tts_val_loss = []
        model.eval()
        tts_model.eval()
        times_val = []
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

                # Compute loss for acoustic model
                target_p = f0
                target_e = energy
                log_D = torch.log(duration_target + hp.log_offset)
                d_loss, m_loss, sum_loss, post_p_loss, post_e_loss, _ =\
                tts_loss(log_duration_output, log_D, target_p, post_p, 
                        target_e, post_e,
                    mel_predicted, mel, ~src_mask, ~mel_mask, iteration, 
                target_p_pho, target_e_pho, None)
                
                FSloss, KLloss = align_criterion(src_len, mel_len, soft_A, hard_A, perf_mask)
                g_loss = guide_loss(soft_A, src_len, mel_len)
                a_loss = FSloss+hp.kl_lamb*KLloss

                total_tts_loss = g_loss + d_loss + 5*m_loss + 10*a_loss + sum_loss + 5*post_p_loss + 5*post_e_loss
                
                tts_val_loss.append(m_loss.item())
                train_subject = "_".join(file_name[0].split("_")[:-1])
                condition_subject = train_subject
                start = time.time()
                if condition_subject in train_subjects_list:
                    iter = train_subjects_list.index(condition_subject)
                    one_hot = one_hot_all[:,iter,:]
                    loss, v_pred_vert = \
                    model(up_emb, pho_label, hard_A, template, vertice, one_hot, criterion, args.up_pho)
                else:
                    for iter in range(one_hot_all.shape[-1]):
                        condition_subject = train_subjects_list[iter]
                        one_hot = one_hot_all[:,iter,:]
                        loss, v_pred_vert = \
                        model(up_emb, pho_label, hard_A, template, vertice, one_hot, criterion)
                end = time.time()
                times_val.append(end-start)
                valid_loss_log.append(loss.item())
            print('validation time:{}'.format(np.sum(times_val)))
            current_loss = np.mean(valid_loss_log)
            
            tts_val_loss = np.mean(tts_val_loss)
            if (e > 0 and e % 25 == 0) or e == args.max_epoch:
                save_checkpoint(model, optimizer, e, save_path)
            if current_loss < min_loss and e > 50:
                save_checkpoint(model, optimizer, e, save_path)
                min_loss = current_loss

            print("epcoh: {}, vertice loss:{:.9f} mel loss: {:.4f}".format(e+1, np.mean(valid_loss_log),tts_val_loss))   
            writer.add_loss(current_loss, iteration, 'Val/avatar')
            writer.add_loss(tts_val_loss, iteration, 'Val/mel')
        
        steplr.step()
    return model, tts_model

         
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description='Avatron')
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
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
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
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
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument('--up_pho', default=True)

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
    # loss
    criterion = nn.MSELoss()

    # Train the model
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr)
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    model, tts_model = \
    trainer(args, dataset["train"], dataset["valid"], model, tts_model, optimizer, steplr, criterion, epoch=args.max_epoch)
    
if __name__=="__main__":
    main()
