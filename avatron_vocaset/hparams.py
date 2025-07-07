import os
seed=1234
batch_size = 12
learning_rate = 0.0001
adam_b1 = 0.5
adam_b2 = 0.9
lr_decay = 0.999
num_workers=0
kl_lamb = 0.001

# Dataset
dataset = "vocaset"

# Text
text_cleaners = ['english_cleaners']

# Audio and mel
sampling_rate = 16000
filter_length = 1024
hop_length = 200
win_length = 800

max_wav_value = 32768.0
n_mel_channels = 80
mel_fmin = 50.0
mel_fmax = 7200.0
mel_dim = 80

num_speaker = 200
spk_hidden_dim = 256
num_spk = 4
num_emo = 4

ref_enc_filters = [32,32,64,64,128,128]
E = 256

encoder_layer = 4
encoder_head = 4
encoder_hidden = 256
decoder_layer = 4
decoder_head = 4
decoder_hidden = 256
fft_conv1d_filter_size = 1024
fft_conv1d_kernel_size = (9, 1)
encoder_dropout = 0.2
decoder_dropout = 0.2
num_group = 2

variance_predictor_filter_size = 256
variance_predictor_kernel_size = 3
variance_predictor_dropout = 0.5

max_seq_len = 1000
min_seq_len = 25

f0_max = 769.0
energy_max = 197.0

lu_conv_kernel_size = 3
lu_conv_output_size = 8
lu_conv_dropout = 0
lu_output_size_w = 16
lu_output_size_c = 2

# Optimizer
n_warm_up_step = 4000

# Log-scaled duration
log_offset = 1.
start_stage = 50000
move_stage= 100000

vae_hidden = 256
anneal_k = 0.0025
anneal_x0 = 50000
anneal_upper= 0.2
