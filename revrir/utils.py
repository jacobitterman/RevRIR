import numpy as np
import scipy.signal as ss
import librosa
import rir_generator as rir  # pip install rir-generator
import os
import matplotlib.pyplot as plt
from functools import partial

HOME = "/user/"


def encode_buff():
    raise NotImplementedError

def decode_buff():
    raise NotImplementedError

def code_logger_wrapper():
    raise NotImplementedError

def MultiBinary():
    raise NotImplementedError

def IndexedBinary():
    raise NotImplementedError


def pad_array(in_arr, max_pad):
    if in_arr.size < max_pad:
        in_arr = np.hstack((in_arr, np.zeros(shape=(max_pad-in_arr.size), dtype=in_arr.dtype)))
    else:
        in_arr = in_arr[:max_pad]
    return in_arr


def rms(x):
    return np.sqrt(np.mean(np.square(x)))


def makedirs_p(path, mode=0o775, LOG=None):
    if not os.path.isdir(path):
        original_umask = os.umask(0)
        try:
            os.makedirs(path, mode, exist_ok=True)
        except Exception as e:
            if LOG:
                LOG.warning(f"Error in dir creation {path}: {e}")
        finally:
            os.umask(original_umask)
    return os.path.exists(path)


def play_audio_ffplay(audio, fs):
    import subprocess
    ffplay_command = ['ffplay', '-f', 'f32le', '-ar', f'{fs}', '-nodisp', '-autoexit', '-']
    ffplay_process = subprocess.Popen(ffplay_command, stdin=subprocess.PIPE)
    ffplay_process.stdin.write(audio.tobytes())
    ffplay_process.stdin.close()
    ffplay_process.wait()


def get_audio(audio_path, fs):
    return librosa.load(audio_path, sr=fs)[0]


def get_audio_paths(dataset='libri-dev-other'):
    if dataset.startswith("libri"):
        libri_files_txt_file_path = f"/{HOME}/asr/scratch/carir/audio_data/libri/LibriSpeech/"
        if dataset == 'libri-dev-other':
            libri_files_txt_file_path = libri_files_txt_file_path + 'DEV_OTHER_FILES.txt'
        elif dataset == 'libri-dev-clean':
            libri_files_txt_file_path = libri_files_txt_file_path + 'DEV_CLEAN_FILES.txt'
        elif dataset == 'libri-train-100':
            libri_files_txt_file_path = libri_files_txt_file_path + 'TRAIN_FILES.txt'
        files_txt_file = libri_files_txt_file_path
        with open(files_txt_file, 'rt') as fid:
            audio_files_path = [x.strip() for x in fid.readlines()]
        print(f'there are {len(audio_files_path)} files')
    elif dataset == 'openSlr_train':
        p = f"/{HOME}/asr/scratch/carir/audio_data/open_slr"
        audio_files_path = [os.path.join(p, a_p) for a_p in os.listdir(p)]
    elif dataset == 'test':
        audio_files_path = [os.path.join(os.path.dirname(__file__), "unit_test", "700-122867-0000.flac"),
                            os.path.join(os.path.dirname(__file__), "unit_test", "700-122867-0003.flac"),
                            os.path.join(os.path.dirname(__file__), "unit_test", "700-122867-0007.flac"),
                            ]
    else:
        raise ValueError(f'unknown dataset: {dataset}')
    return audio_files_path


def get_generater_rirs_paths(mode="train"):
    if mode == "train":
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/generated_rirs/generated_rirs_3K_v2"
    elif mode == "ace":
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/benchmarks/room_classifier/ace/extracted_rirs_normalized"
    elif mode == "test":
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/generated_rirs/generated_rirs_test_300_v2"
    elif mode in ["train_30K",
                  ]:
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/generated_rirs/generated_rirs_{mode}"
    elif mode == "v3_ood_benchmark_3K":
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/generated_rirs/generated_rirs_v3_benchmark_300_ood_rooms"
    elif mode == "v3.1_ood_benchmark_3K":
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/generated_rirs/generated_rirs_v3.1_benchmark_300_ood_rooms"
    elif mode in ["v3.2_train_300K",
                  "v5_benchmark_3K",
                  ]:
        rirs_dir_path = f"/{HOME}/asr/scratch/carir/generated_rirs/rirs_{mode}"
    else:
        assert 0, f"unknown mode {mode}"
    rir_paths = [os.path.join(rirs_dir_path, rir_fp) for rir_fp in sorted(os.listdir(rirs_dir_path))[::-1] if rir_fp.endswith(".pkl")]
    md_paths = [p.replace(".pkl", "_md.csv") for p in rir_paths]
    return rir_paths, md_paths


if __name__ == '__main__':
    pass
