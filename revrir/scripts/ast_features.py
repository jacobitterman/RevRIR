from ..utils import get_audio, get_audio_paths
import torchaudio.compliance.kaldi as ta_kaldi
import librosa
import matplotlib.pyplot as plt
from time import time
import numpy as np


def example():
    fs = 8000
    num_mel_bins = 64
    auds = get_audio_paths(dataset='tmp_tal')[:20]
    audios = np.array([get_audio(aud, fs)[:int(3.5*fs)] for aud in auds])

    n_fft = 1024
    n_hop = int(0.02 * fs)
    tic = time()
    out = [librosa.power_to_db(librosa.feature.melspectrogram(y=a,
                                                              sr=fs,
                                                              n_fft=n_fft,
                                                              hop_length=n_hop,
                                                              win_length=n_fft,
                                                              n_mels=num_mel_bins)) for a in audios]
    print(f'took {time() - tic}')
    import torch
    tic = time()
    fbank = [ta_kaldi.fbank(
        torch.Tensor(audio[None,:]),
        htk_compat=True,
        sample_frequency=fs,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=num_mel_bins,
        dither=0.0,
        frame_length = 128.0,
        frame_shift = 20.0,
        snip_edges=False,
    ) for audio in audios]
    print(f'took {time() - tic}')
    return np.array(out), np.array([a.cpu().numpy() for a in fbank])