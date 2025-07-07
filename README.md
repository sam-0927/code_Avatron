## Setup Dataset 
To train the TTS model, we extract fundamental frequency, energy, and mel-spectrogram from a waveform file. We trim the waveform file to remove a silence range for effecient training.
(vocaset/tts_preprocess.py)

## TTS Training
vocaset/tts_train.py

## Avatar Decoder Training
vocaset/main.py and biwi/main.py
