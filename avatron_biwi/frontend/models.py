import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from transformer.Models import Encoder, Decoder, Encoder2, ProsodyEncoder
from .modules import VarianceAdaptor
from utils_pas import get_mask_from_lengths, get_mask_from_matrix, get_mask_residual
import hparams as hp
import pdb
from .style import Style_Encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrontEnd(nn.Module):
    def __init__(self):
        super(FrontEnd, self).__init__()
        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()
        self.decoder = Decoder()
        
        self.style_encoder = Style_Encoder(hp)
        self.pitch_encoder = ProsodyEncoder()
        self.energy_encoder = ProsodyEncoder()
        
    def forward(self, spk_emb, emo_mel, src_seq, src_len, ref_pitch, ref_energy, mel,  
        		mel_len=None, max_src_len=None, max_mel_len=None):

        # Get text embedding.
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        encoder_output, enc_attns = self.encoder(src_seq, src_mask, True)
        attn_mask = mel_mask.unsqueeze(1).expand(-1, encoder_output.size(1), -1) \
		    if mel_mask is not None else None   
        
        # Get prosody spk embedding
        if spk_emb is None:
            spk_emb = self.style_encoder(emo_mel)
            spk_emb = spk_emb.unsqueeze(1).expand(-1,encoder_output.size(1),-1)
        else:
            spk_emb = spk_emb.unsqueeze(0).unsqueeze(1).expand(-1,encoder_output.size(1),-1)

        # Inputs
        text_pro_input = encoder_output + spk_emb

        # Get Alignment
        soft_A, hard_A, perf_mask, duration_target \
        = self.variance_adaptor(text_pro_input, src_mask, src_len, attn_mask, 
				mel_len, mel, mel_mask, max_mel_len)
        
        # Get pho-level pitch and energy emb.
        ref_pitch_pho = torch.matmul(hard_A, ref_pitch.unsqueeze(2))
        ref_pitch_pho = ref_pitch_pho.squeeze(2) / duration_target 
        ref_pitch_pho = ref_pitch_pho.masked_fill(src_mask, 0)
        ref_pitch_emb = self.variance_adaptor.pitch_emb(ref_pitch_pho.unsqueeze(2))
        ref_energy_pho = torch.matmul(hard_A, ref_energy.unsqueeze(2))
        ref_energy_pho = ref_energy_pho.squeeze(2) / duration_target 
        ref_energy_pho = ref_energy_pho.masked_fill(src_mask, 0)
        ref_energy_emb = self.variance_adaptor.energy_emb(ref_energy_pho.unsqueeze(2))

        # Upsampling 
        upsample_emb = encoder_output + spk_emb + ref_pitch_emb + ref_energy_emb
        output = torch.matmul(hard_A.transpose(1,2), upsample_emb)
        up_text_emb = torch.matmul(hard_A.transpose(1,2), encoder_output)
      
        # Train duration predictor. (infer.)      
        d_prediction = self.variance_adaptor.duration_predictor(text_pro_input.detach(), src_mask) 
        
        # Post prosody (pitch, energy)
        post_pitch = self.variance_adaptor.pitch_decoder(text_pro_input, src_mask)
        post_energy = self.variance_adaptor.energy_decoder(text_pro_input, src_mask)
        
        # Decoder 
        mel_output = self.decoder(output, mel_mask)
        
        return up_text_emb, mel_output, d_prediction, src_mask, mel_mask, mel_len,\
               enc_attns, \
               soft_A, hard_A, duration_target, perf_mask, \
	       post_pitch, post_energy, ref_pitch_pho, ref_energy_pho
    
    def validation(self, spk_emb, emo_mel, step, src_seq, src_len, ref_pitch, ref_energy, mel,  
        		mel_len=None, max_src_len=None, max_mel_len=None):

        # Get text embedding.
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        encoder_output, enc_attns = self.encoder(src_seq, src_mask, True)
        attn_mask = mel_mask.unsqueeze(1).expand(-1, encoder_output.size(1), -1) \
		    if mel_mask is not None else None   
        
        # Get spk embedding
        if spk_emb is None:
            spk_emb = self.style_encoder(emo_mel)
            spk_emb = spk_emb.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
        else:
            spk_emb = spk_emb.unsqueeze(0).unsqueeze(1).expand(-1,encoder_output.size(1),-1)

 
        # Inputs
        text_pro_input = encoder_output + spk_emb 

        # Get Alignment
        soft_A, hard_A, perf_mask, duration_target \
        = self.variance_adaptor(text_pro_input, src_mask, src_len, attn_mask, 
				mel_len, mel, mel_mask, max_mel_len)
        
        # Get pho-level pitch and energy emb.
        ref_pitch_pho = torch.matmul(hard_A, ref_pitch.unsqueeze(2))
        ref_pitch_pho = ref_pitch_pho.squeeze(2) / duration_target 
        ref_pitch_pho = ref_pitch_pho.masked_fill(src_mask, 0)
        ref_pitch_emb = self.variance_adaptor.pitch_emb(ref_pitch_pho.unsqueeze(2))
        ref_energy_pho = torch.matmul(hard_A, ref_energy.unsqueeze(2))
        ref_energy_pho = ref_energy_pho.squeeze(2) / duration_target 
        ref_energy_pho = ref_energy_pho.masked_fill(src_mask, 0)
        ref_energy_emb = self.variance_adaptor.energy_emb(ref_energy_pho.unsqueeze(2))
       
        # Upsampling 
        upsample_emb = encoder_output + spk_emb + ref_pitch_emb + ref_energy_emb
        output = torch.matmul(hard_A.transpose(1,2), upsample_emb)

        # Decoder 
        mel_output = self.decoder(output, mel_mask)
        
        # Train duration predictor. (infer.)      
        d_prediction = self.variance_adaptor.duration_predictor(text_pro_input.detach(), src_mask) 
        
        # Post prosody (pitch, energy)
        post_pitch = self.variance_adaptor.pitch_decoder(text_pro_input, src_mask)
        post_energy = self.variance_adaptor.energy_decoder(text_pro_input, src_mask)
        post_pitch_emb = self.variance_adaptor.pitch_emb(post_pitch.unsqueeze(2))
        post_energy_emb = self.variance_adaptor.energy_emb(post_energy.unsqueeze(2))
        
        # Pred decoder
        output_pro = encoder_output + spk_emb + post_pitch_emb + post_energy_emb
        output_pro = torch.matmul(hard_A.transpose(1,2), output_pro) 
        up_text_emb = torch.matmul(hard_A.transpose(1,2), encoder_output) 

        mel_output_pro = self.decoder(output_pro, mel_mask)
            
        # For samples in tensorboard
        v_upsample_emb = text_pro_input + post_pitch_emb + post_energy_emb
        v_output, v_mel_len, v_mel_mask = self.variance_adaptor.inference(v_upsample_emb, d_prediction)

        v_mel_output = None
        if step > hp.move_stage:
            v_mel_output = self.decoder(v_output, v_mel_mask)

        return up_text_emb, mel_output, d_prediction, src_mask, mel_mask, mel_len,\
               enc_attns,\
               soft_A, hard_A, duration_target, perf_mask, \
               v_mel_output, post_pitch, post_energy, mel_output_pro,\
               ref_pitch_pho, ref_energy_pho
 
    def inference(self, emo_mel, src_seq, src_len, mel_len, max_mel_len, spk_emb=None):
        # Get text embedding.
        src_mask = get_mask_from_lengths(src_len, None)
        encoder_output, enc_attns = self.encoder(src_seq, src_mask, True)
        
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
       
        # Get emotion embedding
        if spk_emb is None:
          spk_emb = self.style_encoder(emo_mel)
        if len(spk_emb.size()) == 1:
          spk_emb = spk_emb.unsqueeze(0)
        spk_emb = spk_emb.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
 
        # Inputs
        text_pro_input = encoder_output + spk_emb
        d_prediction = self.variance_adaptor.duration_predictor(text_pro_input, src_mask) 
        
        # Post prosody (pitch, energy)
        post_pitch = self.variance_adaptor.pitch_decoder(text_pro_input, src_mask)
        post_energy = self.variance_adaptor.energy_decoder(text_pro_input, src_mask)
        post_pitch_emb = self.variance_adaptor.pitch_emb(post_pitch.unsqueeze(2))
        post_energy_emb = self.variance_adaptor.energy_emb(post_energy.unsqueeze(2))
        
        # Upsample
        x = text_pro_input + post_pitch_emb + post_energy_emb
        variance_adaptor_output, _, mel_mask = self.variance_adaptor.inference(x, d_prediction)
        up_text_emb,_,_ = self.variance_adaptor.inference(encoder_output, d_prediction)

        # Decoder 
        mel_output = self.decoder(variance_adaptor_output, mel_mask)
 
        return mel_output, d_prediction, up_text_emb

    def inference_upemb(self, emo_mel, src_seq, src_len, mel_len, max_mel_len, spk_emb=None):
        # Get text embedding.
        src_mask = get_mask_from_lengths(src_len, None)
        encoder_output, enc_attns = self.encoder(src_seq, src_mask, True)

        # Get emotion embedding
        if spk_emb is None:
          spk_emb = self.style_encoder(emo_mel)
        if len(spk_emb.size()) == 1:
          spk_emb = spk_emb.unsqueeze(0)
        spk_emb = spk_emb.unsqueeze(1).expand(-1, encoder_output.size(1), -1)
 
        # Inputs
        text_pro_input = encoder_output + spk_emb
        d_prediction = self.variance_adaptor.duration_predictor(text_pro_input, src_mask) 
        up_text_emb,_,_ = self.variance_adaptor.inference(encoder_output, d_prediction)
        return up_text_emb
        
       

