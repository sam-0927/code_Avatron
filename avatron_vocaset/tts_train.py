import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch.nn as nn
import random
import torch
import torch.nn.functional as F
import soundfile as sf
import scipy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from utils import get_dataset_filelist, plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint,\
scan_checkpoint_tts, plot_alignment, plot_attn, v_plot_attn, count_parameters 
from align_loss import AlignLoss

torch.backends.cudnn.benchmark = True

########### TTS #############
from frontend.models import FrontEnd
from frontend.loss import Loss, guide_loss
from transformer.Models import Decoder
import hparams as hp
import utils_pas
import numpy as np
import pdb
############################


def get_filelist(a):
    file_path = a.input_training_file
    dataset_dict={}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            path, _, spk = lines[i].strip().split('|')
            if 'libritts' in path:
                name = path.split('/')[-1][:-4] # 3630_11612_000019_000001
                key = str(spk)
                new_path = os.path.join('/'.join(path.split('/')[:-2]),'mel','libritts-mel-{}.npy'.format(name))
            elif 'vctk' in path:
                name = path.split('/')[-1][:-4] # p376_291
                key = str(spk) # 376
                new_path = os.path.join('/'.join(path.split('/')[:-2]),'mel','vctk-mel-{}.npy'.format(name))
            elif 'vocaset' in path:
                name = path.split('/')[-1][:-4] # FaceTalk_170915_00223_TA_sentence40_enhanced
                key = str(spk) 
                new_path = os.path.join('/'.join(path.split('/')[:-1]), 'mel', 
                                        'vocaset-mel-{}.npy'.format(name))
            elif 'BIWI' in path:
                name = path.split('/')[-1][:-4] # F1_01
                key = str(spk)          
                new_path = os.path.join('/'.join(path.split('/')[:-2]), 'mel', 
                                        'biwi-mel-{}.npy'.format(name))
            else:
                print('Unknow wav path!')
                pdb.set_trace()
            if key not in dataset_dict:
                dataset_dict[key] = [new_path]
            else:
                dataset_dict[key].append(new_path)
    return dataset_dict

def get_style_mel(dataset_dict, spk):
    mels, mel_paths = [], []
    for i in range(len(spk)):
        key = str(spk[i])
        idx = random.randint(0, len(dataset_dict[key])-1)
        mel_path = dataset_dict[key][idx]
        mel_paths.append(mel_path)
        mel = torch.from_numpy(np.load(mel_path))
        mels.append(mel)
    mels = utils_pas.pad_2D(mels)
    mels = torch.from_numpy(mels)
    return mels, mel_paths

def train(rank, a):
    torch.cuda.manual_seed(hp.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model_tts = FrontEnd().to(device) # TTS on device

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')

    steps = 0
    if cp_g is None:
        last_epoch = -1
        print('checkpoint not found. begin from 0')
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        model_tts.load_state_dict(state_dict_g['model'], strict=False)
        steps = state_dict_g['steps'] + 1
        last_epoch = state_dict_g['epoch']
        print('checkpoint loaded from ', a.checkpoint_path)

    optim_g = torch.optim.AdamW(model_tts.parameters()
                                , hp.learning_rate, betas=[hp.adam_b1, hp.adam_b2])
   
    if cp_g is not None:
        optim_g.load_state_dict(state_dict_g['optim_g'])
        
    torch.nn.utils.clip_grad_norm_(model_tts.parameters(), 1.0) # TTS  

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.lr_decay, last_epoch=-1)

    tts_loss = Loss().to(device)
    align_criterion = AlignLoss().to(device)

    # Load dataset.
    training_data, validation_data = get_dataset_filelist(a)
    train_loader = DataLoader(training_data, num_workers=hp.num_workers, shuffle=True, 
                              sampler=None, collate_fn=training_data.collate_fn, 
                              batch_size=hp.batch_size, pin_memory=True, drop_last=True)

    if rank == 0: 
        validation_loader = DataLoader(validation_data, num_workers=1, shuffle=True, 
                                       sampler=None, collate_fn=validation_data.collate_fn, 
                                       batch_size=hp.batch_size, pin_memory=True, drop_last=True)
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    # training
    model_tts.train()
    min_loss = 1000
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        
        print('len train loader:', len(train_loader))
        for i, batch in enumerate(train_loader):
            if rank == 0: start_b = time.time()
            tts_data = batch

            ### for TTS data
            name = tts_data['id']
            spk = tts_data['speaker']
            text = torch.from_numpy(tts_data['text']).long().to(device)
            mel_target = torch.from_numpy(tts_data["mel_target"]).float().to(device) 
            f0 = torch.from_numpy(tts_data['f0']).float().to(device) 
            f0 = f0/hp.f0_max
            
            energy = torch.from_numpy(tts_data['energy']).float().to(device) 
            energy = energy/hp.energy_max
            src_len = torch.from_numpy(tts_data['src_len']).long().to(device)
            mel_len = torch.from_numpy(tts_data['mel_len']).long().to(device) 
            max_src_len = np.max(tts_data['src_len']).astype(np.int32)
            max_mel_len = np.max(tts_data['mel_len']).astype(np.int32)
            if f0.size(1) != max_mel_len:
                f0 = f0[:,:max_mel_len]
            dataset_dict = get_filelist(a)
            emo_mel, emo_mel_path = get_style_mel(dataset_dict, spk)
            emo_mel = emo_mel.float().to(device)
            ###### TTS forward
            _, mel_predicted, log_duration_output, src_mask, mel_mask, _,\
            enc_attns, \
            soft_A, hard_A, duration_target, perf_mask,\
            post_p, post_e, target_p_pho, target_e_pho \
            = model_tts(emo_mel, text, src_len, f0, energy,\
	    		mel_target, mel_len, max_src_len, max_mel_len)

            #! Generator
            optim_g.zero_grad()

            target_p = f0
            target_e = energy
            log_D = torch.log(duration_target + hp.log_offset)
            d_loss, m_loss, sum_loss, post_p_loss, post_e_loss, _ =\
            tts_loss(log_duration_output, log_D, target_p, post_p, 
                     target_e, post_e,
        	     mel_predicted, mel_target, ~src_mask, ~mel_mask, steps, 
		     target_p_pho, target_e_pho, None)
            
            FSloss, KLloss = align_criterion(src_len, mel_len, soft_A, hard_A, perf_mask)
            g_loss = guide_loss(soft_A, src_len, mel_len)

            a_loss = FSloss+hp.kl_lamb*KLloss

            # Total TTS loss
            if steps > hp.move_stage:
                total_tts_loss = g_loss + d_loss + 5*m_loss + 10*a_loss + sum_loss + 5*post_p_loss + 5*post_e_loss
            if steps <= hp.move_stage:
                total_tts_loss = g_loss + 5*m_loss + 10*a_loss + 5*post_p_loss + 5*post_e_loss
            if torch.isnan(total_tts_loss):
                pdb.set_trace()
            t_l = total_tts_loss.item()
            d_l = d_loss.item()
            m_l = m_loss.item()
            a_l = a_loss.item()
            sum_l = sum_loss.item()
            post_p_l = post_p_loss.item()
            post_e_l = post_e_loss.item()

            loss_gen_all = total_tts_loss

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        print('GAN Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                              format(steps, loss_gen_all, m_l, time.time() - start_b))

                    print('TTS steps : Total loss: {:.4f}, align loss: {:.4f}, Dur loss: {:.4f}, sum: {:.4f}, post F0: {:.4f}, post E: {:.4f}'.
                          format(t_l, a_l, d_l, sum_l, post_p_l, post_e_l))
                    print()
                
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0: #* 50000 step마다 저장
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    { 'model': model_tts.state_dict(),
                                     'optim_g': optim_g.state_dict(), 
                                     'steps': steps,
                                     'epoch': epoch})
                
                # Tensorboard summary logging
                if steps % a.summary_interval == 0: #* 500
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/duration_error", d_loss, steps)
                    sw.add_scalar("training/post_energy_error", post_e_loss, steps)
                    sw.add_scalar("training/post_pitch_error", post_p_loss, steps)
                    sw.add_scalar("training/mel_loss", m_loss, steps)
                    sw.add_scalar("training/align_loss", a_loss, steps)
                    sw.add_scalar("training/guide_loss", g_loss, steps)
                    sw.add_scalar("training/sum_loss", sum_loss, steps)
                    sw.add_scalar("training/kl_loss", KLloss, steps)

                    plot_attn(sw, enc_attns, soft_A, hard_A, steps, hp)
                # Validation
                if steps % a.validation_interval == 0:  #and steps != 0:
                    model_tts.eval()
                    val_err_tot = 0
                    with torch.no_grad():
                        total_valid_error = 0
                        for j, batch in enumerate(validation_loader):
                            print(len(validation_loader), j)
                            val_tts_data = batch
                            
                            ##########
                            v_id = val_tts_data['id']
                            v_spk = val_tts_data['speaker']
                            v_text = torch.from_numpy(val_tts_data['text']).long().to(device)
                            v_mel_target = torch.from_numpy(val_tts_data["mel_target"]).float().to(device)
                            v_f0 = torch.from_numpy(val_tts_data['f0']).float().to(device)
                            v_f0 = v_f0/hp.f0_max
                            v_energy = torch.from_numpy(val_tts_data['energy']).float().to(device)
                            v_energy = v_energy/hp.energy_max
                            v_src_len = torch.from_numpy(val_tts_data['src_len']).long().to(device)
                            v_mel_len = torch.from_numpy(val_tts_data['mel_len']).long().to(device)
                            v_max_src_len = np.max(val_tts_data['src_len']).astype(np.int32)
                            v_max_mel_len = np.max(val_tts_data['mel_len']).astype(np.int32)
                            if v_f0.size(1) != v_max_mel_len:
                                v_f0 = v_f0[:,:v_max_mel_len]

                            v_emo_mel, v_emo_mel_path = get_style_mel(dataset_dict, v_spk)
                            v_emo_mel = v_emo_mel.float().to(device)
                            ss = time.time()
                            _, v_mel_predicted, v_log_duration_output, v_src_mask, \
                            v_mel_mask, v_out_mel_len, v_enc_attns, \
                            v_soft_A, v_hard_A, v_duration_target, v_perf_mask, \
                            v_inf_mel, v_post_p, v_post_e, v_mel_pred_pro,\
			                v_p_target_pho, v_e_target_pho\
                            = model_tts.validation(None, v_emo_mel, steps, 
                                                   v_text, v_src_len, v_f0, v_energy,
                                                   v_mel_target, 
                                                   v_mel_len, v_max_src_len, v_max_mel_len)

                            v_log_D = torch.log(v_duration_target + hp.log_offset)

                            v_d_loss, v_m_loss, v_sum_loss, v_post_p_loss, v_post_e_loss, v_m_loss_pro \
                            = tts_loss(v_log_duration_output, v_log_D, v_f0, v_post_p, 
                                       v_energy, v_post_e,
                                       v_mel_predicted, v_mel_target, 
                                       ~v_src_mask, ~v_mel_mask, steps, 
				       v_p_target_pho, v_e_target_pho, v_mel_pred_pro)
                            total_valid_error += v_m_loss
                            if j <= 4:
                                v_plot_attn(sw, v_soft_A, v_hard_A, steps, hp, j)
                                sw.add_scalar("validation/mel_spec_error_{}".format(j), 
                               				  v_m_loss, steps)                               
                                sw.add_scalar("validation/duration_{}".format(j), 
                                              v_d_loss, steps)
                                sw.add_scalar("validation/sum_{}".format(j), 
                                              v_sum_loss, steps)
                                sw.add_scalar("validation/post_p_{}".format(j), 
                                              v_post_p_loss, steps)
                                sw.add_scalar("validation/post_e_{}".format(j), 
                                              v_post_e_loss, steps)

                                sw.add_figure('generated/ref_spec_{}'.format(j), 
                				plot_spectrogram(v_mel_target[0].squeeze(0).cpu().numpy().transpose(1,0)), steps) 
                                sw.add_figure('generated/pred_spec_{}'.format(j), 
                			    plot_spectrogram(v_mel_predicted[0].squeeze(0).cpu().numpy().transpose(1,0)), steps)

                                if steps > hp.move_stage:
                                    sw.add_figure('generated/inference_spec_{}'.format(j), 
                                    plot_spectrogram(v_inf_mel[0].squeeze(0).cpu().numpy().transpose(1,0)), steps)
                        total_valid_error /= len(validation_loader)
                        sw.add_scalar("validation/total_valid_error", 
                                      total_valid_error, steps)
                    # checkpointing
                    if total_valid_error < min_loss:
                        min_loss = total_valid_error 
                        checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                        save_checkpoint(checkpoint_path,
                                        { 'model': model_tts.state_dict(),
                                         'optim_g': optim_g.state_dict(), 
                                         'steps': steps,
                                         'epoch': epoch})

                    model_tts.train()

            steps += 1
        scheduler_g.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, 
                                                    int(time.time() - start)))

def main():
    print('Initializing Training Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_training_file', default='filelists/libritts_vctk/train.txt')
    parser.add_argument('--input_validation_file', default='filelists/libritts_vctk/valid.txt')
    parser.add_argument('--checkpoint_path', '-c', default='./log_and_save/results_vctk')
    parser.add_argument('--config', default='config_vp.json')
    parser.add_argument('--training_epochs', default=1000000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=5000, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    a = parser.parse_args()

    torch.manual_seed(hp.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hp.seed)
        hp.num_gpus = torch.cuda.device_count()
        print(hp.num_gpus)
        hp.batch_size = int(hp.batch_size / hp.num_gpus)
        print('Batch size per GPU :', hp.batch_size)
    else: pass

    if hp.num_gpus > 1:
        mp.spawn(train, nprocs=hp.num_gpus, args=(a,))
    else:
        train(0, a)

if __name__ == '__main__':
    main()
