import os
from torch.utils.tensorboard import SummaryWriter
from .plot_image import *

def get_writer(log_directory):
    writer = TTSWriter(log_directory)
            
    return writer


class TTSWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(TTSWriter, self).__init__(log_dir)
        
    def add_loss(self, loss, global_step, phase):
        self.add_scalar(f'{phase}_loss', loss, global_step)
    
    def add_specs(self, mel_padded, mel_out, mel_out_post, mel_lengths, global_step, phase):
        mel_fig = plot_melspec(mel_padded, mel_out, mel_out_post, mel_lengths)
        self.add_figure(f'{phase}_melspec', mel_fig, global_step)
        
    def add_alignments(self, enc_dec_alignments,
                       text_padded, mel_lengths, text_lengths, global_step, phase):

        enc_dec_align_fig = plot_alignments(enc_dec_alignments, text_padded, mel_lengths, text_lengths, 'enc_dec')
        self.add_figure(f'{phase}_enc_dec_alignments', enc_dec_align_fig, global_step)
    
    def add_alignment(self, enc_dec_alignments, mel_lengths, text_lengths, global_step, phase):

        enc_dec_align_fig = plot_align(enc_dec_alignments, text_lengths, mel_lengths)
        self.add_figure(f'{phase}_enc_dec_alignments', enc_dec_align_fig, global_step)
       
    def add_gates(self, gate_out, global_step, phase):
        gate_fig = plot_gate(gate_out)
        self.add_figure(f'{phase}_gate_out', gate_fig, global_step)
