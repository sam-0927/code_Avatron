import numpy as np
import os
from scipy.io.wavfile import read
from scipy.io.wavfile import write as write_wav
import pyworld as pw
import torch
import audio as Audio
from utils_pas import get_alignment
from utils_tts import standard_norm, remove_outlier, average_by_duration
import hparams as hp
import codecs
import librosa
import pdb
import sys
from string import punctuation
import re
from g2p_en import G2p

from sklearn.preprocessing import StandardScaler
g2p = G2p()

from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def get_torch_mel(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    y = torch.nn.functional.pad(y.unsqueeze(1), (int(n_fft/2), int(n_fft/2)), mode='reflect')
   
    y = y.squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, 
                      window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    mag = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))
    energy = torch.norm(mag, dim=1)

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], mag)
    spec = spectral_normalize_torch(spec)
    return spec, energy

newf = open('dataset_path.txt', 'w')
def build_from_path(out_dir, meta):
    total_duration = 0
    with open(meta, encoding='utf-8') as f:
        for index, line in enumerate(f):
            path, text, spk = line.strip().split('|')
            
            basename = path.split('/')[-1][:-4]
            ret = process_utterance(text, out_dir, path, basename, spk)

            if ret is None:
                continue
            else:
                line, wav_length = ret
                total_duration += wav_length / 16000
                newf.write(line+'\n')
    
    hour = total_duration // 3600
    mini = (total_duration - hour*3600) / 60
    print('Total duration after preprocessing train-clean-360: {:.0f} hour {:.1f} min'.format(hour, mini))

def rescale(x):
    ori = 10*np.log10(np.mean(np.square(x)))
    alpha = np.sqrt(np.power(10, (ori-3)/10)*len(x)/np.sum(np.square(x)))
    return alpha*x

def save_wav(name, x):
    from scipy.io.wavfile import write
    x = x*32768
    write('useless/'+name+'.wav', 16000, x.astype(np.int16))
   
def process_utterance(text, out_dir, path, basename, spk):
    text = text
    wav_path = path
    if os.path.isfile(wav_path) == False:
        return None
        
    # Prepare phoneme
    text = g2p(text)
    text = '{'+ '}{'.join(text) + '}'
    text = text.replace('}{', ' ')
    
    # Read and trim wav files
    wav,_ = librosa.load(wav_path, sr=hp.sampling_rate)
    wav = wav / np.max(np.abs(wav))
    wav = rescale(wav)

    wav = librosa.effects.trim(wav, top_db=23, frame_length=1024, hop_length=256)[0] 
    f0, _ = pw.harvest(wav.astype(np.float64), hp.sampling_rate, frame_period=(hp.hop_length/hp.sampling_rate)*1000)
    
    # Compute mel-scale spectrogram and energy
    mel_spectrogram, energy = get_torch_mel(torch.FloatTensor(wav).unsqueeze(0), 
                                            n_fft=hp.filter_length, 
                                            num_mels=hp.mel_dim, sampling_rate=hp.sampling_rate,
                                            hop_size=hp.hop_length, win_size=hp.win_length, 
                                            fmin=hp.mel_fmin, fmax=hp.mel_fmax)
    mel_spectrogram = mel_spectrogram.squeeze(0)
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)
    energy = energy.squeeze(0)
    energy = energy.numpy().astype(np.float32)
    print('f0:', f0.shape)
    print('energy:', energy.shape)
    print('mel:', mel_spectrogram.shape)
    f0, energy = remove_outlier(f0), remove_outlier(energy)
    if np.sum(f0) == 0:
        #pdb.set_trace()
        save_wav(basename, wav)
        return None

    if mel_spectrogram.shape[1] >= hp.max_seq_len:
        return None
    elif mel_spectrogram.shape[1] <= hp.min_seq_len:
        return None

    # Save wave files
    wav_filename = '{}.wav'.format(basename)
    write_wav(os.path.join(out_dir, 'wavs', wav_filename), hp.sampling_rate, wav)

    # Save fundamental prequency
    f0_filename = '{}-f0-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'f0', f0_filename), f0, allow_pickle=False)

    # Save energy
    energy_filename = '{}-energy-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'energy', energy_filename), energy, allow_pickle=False)

    # Save spectrogram
    mel_filename = '{}-mel-{}.npy'.format(hp.dataset, basename)
    np.save(os.path.join(out_dir, 'mel', mel_filename), mel_spectrogram.T, allow_pickle=False)
    
    return '|'.join([path, text, spk]), wav.shape[0]
