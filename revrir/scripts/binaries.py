import librosa

from ..utils import MultiBinary, IndexedBinary, decode_buff, encode_buff

import numpy as np
from tqdm import tqdm
from rich.progress import Progress
from functools import partial

from ..utils import play_audio_ffplay, get_audio

#TODO remove the script


def generate_16KHz_bin(save_path='/home/talr/Downloads/libri_16k.bin'):
    from CARIR.utils import get_audio, get_audio_paths
    training_dataset = 'libri-dev-other'
    n_samples_training = 2000
    fs = 16000
    audio_paths = get_audio_paths(training_dataset)[:n_samples_training]
    print(f'selected {n_samples_training} audio samples for training, in practice we got {len(audio_paths)}')
    with IndexedBinary(save_path, 'w') as ibin:
        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(audio_paths))
            for audio_path in audio_paths:
                audio = get_audio(audio_path, fs)
                audio_dict = {'audio': audio,
                              'fs': fs}
                buff = encode_buff(audio_dict)
                ibin.write(buff)
                progress.update(task, advance=1)
    print('finished')


def reader_worker(source_path, target_fs):
    decode_fn = partial(decode_buff, decode_audio_dtype=np.float32)
    ibin_r = IndexedBinary(source_path, 'r', getitem_fn=decode_fn)

    def get_buff(st_indx, end_indx, dst_path):
        with IndexedBinary(dst_path, 'w') as ibin_w:
            for i in tqdm(range(st_indx, end_indx), total = end_indx - st_indx):
                a = ibin_r[i]
                if 'audio' in a:
                    audio = librosa.resample(a['audio'], orig_sr=a['fs'], target_sr=target_fs)
                elif 'audio_path' in a:
                    audio = get_audio(a['audio_path'], target_fs)
                else:
                    assert 0, a.keys()
                audio_dict = {'audio': audio,
                              'audio_path': a['audio_path'],
                              'duration': a['duration'],
                              'idx': i,
                              'fs': target_fs}
                buff = encode_buff(audio_dict)
                ibin_w.write(buff)
        return f"finished {dst_path}"
    return get_buff


def parallel_convert_bin_to_target_fs(source_path='/orcam/speech3/scratch/transcription/open_slr/english_parsed/binaries_ENGLISH_PYTORCH_V2_filtered_by_csr/train_csr_yt_95_or_open_slr_95_coverage_89.0/segments_up_to_16_sec.bin',
                             dst_path='/orcam/asr/scratch/carir/audio_data/open_slr/train_csr_yt_95_or_open_slr_95_coverage_89.0_segments_up_to_16_sec',
                             target_fs=8000):
    num_workers = 30
    num_samples = 5947546
    chunk_size =  num_samples // num_workers
    init_kwargs = [{'source_path': source_path, 'target_fs': target_fs} for i in range(num_workers)]
    from parallel_tools.par_worker.par_worker import par_map
    kwargs_scatter = [{'st_indx': i * chunk_size, 'end_indx': min((i+1) * chunk_size, num_samples), 'dst_path': dst_path + f"_{i}.ibin"} for i in range(num_workers)]
    par_iter = par_map(reader_worker,
                       init_kwargs= init_kwargs,
                       kwargs_scatter=kwargs_scatter,
                       worker_count=num_workers,
                       timeout_sec=None,
                       raise_exceptions=False,
                       debug=True,
                       hosts=['blade25', 'blade40', 'blade27a'])

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=num_workers)
        for i, res in par_iter:
            print(res)
            progress.update(task, advance=1)


def convert_bin_to_target_fs(source_path='/orcam/speech3/scratch/transcription/open_slr/english_parsed/binaries_ENGLISH_PYTORCH_V2_filtered_by_csr/train_csr_yt_95_or_open_slr_95_coverage_89.0/segments_up_to_16_sec.bin',
                             dst_path='/orcam/asr/scratch/carir/audio_data/open_slr/train_csr_yt_95_or_open_slr_95_coverage_89.0_segments_up_to_16_sec.bin',
                             target_fs=8000):
    decode_fn = partial(decode_buff, decode_audio_dtype=np.float32)
    with IndexedBinary(dst_path, 'w') as ibin_w:
        with IndexedBinary(source_path, 'r', getitem_fn=decode_fn) as ibin_r:
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing...", total=len(ibin_r))
                for a in ibin_r:
                    if 'audio' in a:
                        audio = librosa.resample(a['audio'], orig_sr=a['fs'], target_sr=target_fs)
                    elif 'audio_path' in a:
                        audio = get_audio(a['audio_path'], target_fs)
                    else:
                        assert 0, a.keys()
                    audio_dict = {'audio': audio,
                                  'audio_path': a['audio_path'],
                                  'duration': a['duration'],
                                  'fs': target_fs}
                    buff = encode_buff(audio_dict)
                    ibin_w.write(buff)
                    progress.update(task, advance=1)


def convert_rirs_to_bin(rirs_path='/home/talr/Downloads/rirs_small.pkl',
                        rirs_md_path='/home/talr/Downloads/rirs_small_md.csv',
                        save_path='/home/talr/Downloads/'):
    import pickle as pkl
    import pandas as pd
    import os

    rirs_version, rirs_max_length, rirs = pkl.load(open(rirs_path, 'rb'))
    rirs_max_length = int(rirs_max_length)
    rirs_md = pd.read_csv(rirs_md_path)
    assert len(rirs) == len(rirs_md)

    rirs_ibin_path = os.path.join(save_path, os.path.basename(rirs_path).split('.')[0] + '.bin')
    rirs_md_ibin_path = os.path.join(save_path, os.path.basename(rirs_md_path).split('.')[0] + '.bin')
    with IndexedBinary(rirs_ibin_path, 'w') as ibin_rir:
        with IndexedBinary(rirs_md_ibin_path, 'w') as ibin_rir_md:
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing...", total=len(rirs))
                for rir, rir_md in zip(rirs, rirs_md.iterrows()):
                    row_id, rir_md = rir_md

                    # rir bin
                    rir_dict = {'audio': rir,
                                'fs': rir_md['fs'],
                                'version': rirs_version,
                                'max_length': rirs_max_length}
                    buff_rir = encode_buff(rir_dict)
                    ibin_rir.write(buff_rir)

                    # rir_md bin
                    buff_rir_md = encode_buff(rir_md)
                    ibin_rir_md.write(buff_rir_md)

                    # update progress bar
                    progress.update(task, advance=1)


def example_check_16khz_2_8khz(bin_path='/home/talr/Downloads/libri_8k.bin'):
    decode_fn = partial(decode_buff, decode_audio_dtype=np.float32)
    with IndexedBinary(bin_path, 'r', getitem_fn=decode_fn) as ibin_r:
        idx = 10
        a_dict = ibin_r[idx]
        play_audio_ffplay(a_dict['audio'][:5 * a_dict['fs']], a_dict['fs'])
        from IPython import embed; embed()


def example_check_rirs(rirs_path='/home/talr/Downloads/rirs_small.bin',
                       rirs_md_path='/home/talr/Downloads/rirs_small_md.bin', ):
    decode_fn = partial(decode_buff, decode_audio_dtype=np.float32)
    with IndexedBinary(rirs_path, 'r', getitem_fn=decode_fn) as ibin_rir:
        with IndexedBinary(rirs_md_path, 'r', getitem_fn=decode_buff) as ibin_rir_md:
            idx = 0
            rir_md_dict = ibin_rir_md[idx]
            rir_dict = ibin_rir[idx]
            play_audio_ffplay(rir_dict['audio'], rir_dict['fs'])
            from IPython import embed; embed()


if __name__ == '__main__':
    # generate_16KHz_bin()
    parallel_convert_bin_to_target_fs()
    # example_check_16khz_2_8khz()
    # convert_rirs_to_bin()
    # example_check_rirs()
    pass
