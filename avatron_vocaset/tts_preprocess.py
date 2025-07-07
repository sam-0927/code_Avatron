import os
from data import vocaset, vctk, libritts
import hparams as hp

def write_metadata(train, val, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in train:
            f.write(m + '\n')
    with open(os.path.join(out_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        for m in val:
            f.write(m + '\n')

def main():
    in_dir = hp.data_path
    out_dir = hp.preprocessed_path

    mel_out_dir = os.path.join(out_dir, "mel")
    if not os.path.exists(mel_out_dir):
        os.makedirs(mel_out_dir, exist_ok=True)
    f0_out_dir = os.path.join(out_dir, "f0")
    if not os.path.exists(f0_out_dir):
        os.makedirs(f0_out_dir, exist_ok=True)
    energy_out_dir = os.path.join(out_dir, "energy")
    if not os.path.exists(energy_out_dir):
        os.makedirs(energy_out_dir, exist_ok=True)
    wav_out_dir = os.path.join(out_dir, "wavs")
    if not os.path.exists(wav_out_dir):
        os.makedirs(wav_out_dir, exist_ok=True)

    vocaset.build_from_path(out_dir, in_dir)
    vctk.build_from_path(out_dir, in_dir)
    libritts.build_from_path(out_dir, in_dir)
   
if __name__ == "__main__":
    main()
