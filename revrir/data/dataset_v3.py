import os
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from copy import copy

import torch
import scipy.signal as ss
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from .data_utils import get_sim_mat
from ..utils import get_audio, pad_array
from ..eval.classifier_benchmark import name2id


class CARIR_dataset_V3(Dataset):
    def __init__(self,
                 audio_paths,
                 rir_paths,
                 md_paths,
                 fs=8000, # Hz
                 max_audio_length_sec=10,
                 mode: str = "train",
                 hard_negative_prob=1.0,
                 hard_positive_prob=1.0,
                 ):

        self.mode = mode

        # rirs
        if isinstance(rir_paths, str):
            rir_paths = [rir_paths]
        self.rirs = None  # in worker_init_function
        self.rir_paths = rir_paths
        self.rir_path2id = []
        for p in self.rir_paths:
            for k in name2id:
                if k in os.path.basename(p):
                    self.rir_path2id.append(name2id[k])
                    break
        assert len(self.rir_path2id) == len(self.rir_paths), f"one or more of the paths can't be converted into room_id: {self.rir_paths}"

        if isinstance(md_paths, str):
            md_paths = [md_paths]
        self.rirs_md = None  # in worker_init_function
        self.md_paths = md_paths

        self.n_samples = None  # in worker_init_function
        self.n_samples_zero = None  # in worker_init_function
        self.max_rir_length = None  # in worker_init_function
        self.rir_iter = None  # in worker_init_function

        # audios
        self.fs = fs
        self.max_audio_length_sec = max_audio_length_sec
        self.audio_paths = audio_paths
        self.audios = None  # in worker_init_function
        self.len = len(self.audio_paths)

        # sampling
        self.hard_positive_prob = hard_positive_prob
        self.hard_negative_prob = hard_negative_prob
        self.hp_audio_idx_iter = None
        self.rir_iter_hn = None

        print('finished __init__ CARIR dataset')

    def worker_init_function(self, worker_index):
        self.audios = [get_audio(p, self.fs)[:self.max_audio_length_sec * self.fs] for p in
                       tqdm(self.audio_paths, desc=f'loading audios in worker {worker_index}')]
        if worker_index == 0:
            print(f"loading {len(self.rir_paths)} rir files")
        self.rirs = [pkl.load(open(r_p, 'rb')) for r_p in self.rir_paths]
        self.rirs_md = [pd.read_csv(md_p) for md_p in self.md_paths]
        max_rir_length_per_file = None
        if len(self.rirs[0]) == 3:  # each file contain: version, max_rir, rirs
            print(f"reading rir files version {self.rirs[0][0]}")
            max_rir_length_per_file = [r[1] for r in self.rirs]
            self.rirs = [r[2] for r in self.rirs]

        self.n_samples = np.cumsum([len(x) for x in self.rirs])
        self.n_samples_zero = np.hstack((0, self.n_samples[:-1]))
        if max_rir_length_per_file is None:
            self.max_rir_length = np.max([r.shape[-1] for r in self.rirs])
        else:
            self.max_rir_length = np.max(max_rir_length_per_file)
        self.rir_iter = self.rir_itr_fn()
        self.hp_audio_idx_iter = self.hp_audio_idx_iter_fn()
        self.rir_iter_hn = self.rir_itr_fn()

    def hp_audio_idx_iter_fn(self):
        while True:
            for i in np.random.permutation(self.len):
                yield i

    def rir_itr_fn(self):
        rir_epoch = 0
        while True:
            print(f'starting epoch #{rir_epoch} in mode {self.mode}')
            for i in np.random.permutation(self.n_samples[-1]):
                room_id = np.where(i < self.n_samples)[0][0]
                idx_within_room = i - self.n_samples_zero[room_id]
                rir = self.rirs[room_id][idx_within_room]
                rir_md = self.rirs_md[room_id].iloc[idx_within_room]
                yield rir, rir_md, i, self.rir_path2id[room_id]
            rir_epoch += 1

    def __len__(self):
        if self.mode == "train":
            return self.len * 1000
        return self.len

    def __getitem__(self, indx_):
        if self.audios is None:
            self.worker_init_function(0)

        # retrieve audios and audios_path
        indx = indx_ % self.len
        audio = self.audios[indx]
        audio_path = self.audio_paths[indx]

        # retrieve rir
        rir, rir_md, rir_id, room_id = next(self.rir_iter)
        reverebed_audio = ss.convolve(rir, audio)[:self.max_audio_length_sec * self.fs]
        assert (len(audio) <= self.max_audio_length_sec * self.fs and
                len(reverebed_audio) <= self.max_audio_length_sec * self.fs)
        speaker_id = audio_path.rsplit('/',4)[-3]
        book_id = audio_path.rsplit('/',4)[-2]
        data_out = {'audio1': audio,
                    'audio1_idx': indx,
                    'audio1_fpath': audio_path,
                    'speaker_id': speaker_id,
                    'reverbed_audio1': reverebed_audio,
                    'rir_idx': rir_id,
                    'room_id': room_id,
                    'RIR': rir,
                    'rir_md': rir_md.to_dict(),
                    'audio_rir_corr': f'{speaker_id}_{book_id}_{rir_id}',
                    'rir_hn': None,
                    'reverebed_audio_hn': None,
                    'rir_id_hn': None,
                    'rir_md_hn': None,
                    'audio_hp': None,
                    'reverebed_audio_hp': None,
                    'audio_hp_idx': None,
                    'audio_hp_fpath': None,
                    'room_id_hn': None
                    }

        if np.random.rand() <= self.hard_positive_prob:
            indx_hp = copy(indx)
            while indx == indx_hp:
                indx_hp = next(self.hp_audio_idx_iter)
            audio_hp = self.audios[indx_hp]
            reverebed_audio_hp = ss.convolve(rir, audio_hp)[:self.max_audio_length_sec * self.fs]
            data_out['audio_hp'] = audio_hp
            data_out['reverebed_audio_hp'] = reverebed_audio_hp
            data_out['audio_hp_idx'] = indx_hp
            data_out['audio_hp_fpath'] = self.audio_paths[indx_hp]
            assert len(reverebed_audio_hp) <= self.max_audio_length_sec * self.fs

        if np.random.rand() <= self.hard_negative_prob:
            rir_id_hn = copy(rir_id)
            while rir_id == rir_id_hn:
                rir_hn, rir_md_hn, rir_id_hn, room_id_hn = next(self.rir_iter_hn)
            reverebed_audio_hn = ss.convolve(rir_hn, audio)[:self.max_audio_length_sec * self.fs]
            data_out['rir_hn'] = rir_hn
            data_out['reverebed_audio_hn'] = reverebed_audio_hn
            data_out['rir_id_hn'] = rir_id_hn
            data_out['rir_md_hn'] = rir_md_hn
            data_out['room_id_hn'] = room_id_hn

        return data_out


def calc_sim_mat_for_loss(batch_size, rirs_idx, audios_idx, verbose=False, return_tensors='pt'):
    # calc similarity matrix for loss
    sim_mat_gt = np.eye(batch_size)
    sim_mat_rir = get_sim_mat(np.array(rirs_idx, dtype=np.int32), verbose=verbose)
    sim_mat_audio = get_sim_mat(np.array(audios_idx, dtype=np.int32), verbose=verbose)

    # hard positives
    sim_mat_rir = sim_mat_rir - np.eye(len(sim_mat_rir))
    sim_mat_gt += 2 * sim_mat_rir

    # hard negatives
    sim_mat_audio = sim_mat_audio - np.eye(len(sim_mat_audio))
    rs, cs = np.where(sim_mat_audio == 1)
    sim_mat_gt[rs, cs] = -2

    # negatives
    rs, cs = np.where(sim_mat_gt == 0)
    sim_mat_gt[rs, cs] = -1
    if verbose:
        plt.imshow(sim_mat_gt)
        plt.colorbar()
        plt.xticks(range(len(sim_mat_gt)), labels=[f'{x},{y}' for x, y in zip(audios_idx, rirs_idx)], rotation=270)
        plt.yticks(range(len(sim_mat_gt)), labels=[f'{x},{y}' for x, y in zip(audios_idx, rirs_idx)])
        plt.show()

    if return_tensors == 'pt':
        sim_mat_gt = torch.tensor(sim_mat_gt, dtype=torch.int32)
    return sim_mat_gt


def collate_fn_v3(processor=None, fs=8000, nsample_rir=3200, verbose=False):
    def inner_function(batch):
        # list all audio files and ids
        reverbed_audios = []
        audios_idx = []
        rirs_idx = []
        rirs = []
        rirs_md = []
        room_id = []
        extra_kwargs = ["speaker_id", "audio1_fpath", "audio_rir_corr", "RIR"]
        extra_kwargs_flatten = {k: [] for k in extra_kwargs}

        for s in batch:
            reverbed_audios.append(s['reverbed_audio1'])
            audios_idx.append(s['audio1_idx'])
            rirs_idx.append(s['rir_idx'])
            rirs.append(s['RIR'])
            rirs_md.append(s['rir_md'])
            room_id.append(s['room_id'])
            if 'reverebed_audio_hn' in s and s['reverebed_audio_hn'] is not None:
                reverbed_audios.append(s['reverebed_audio_hn'])
                audios_idx.append(s['audio1_idx'])
                rirs_idx.append(s['rir_id_hn'])
                rirs.append(s['rir_hn'])
                rirs_md.append(s['rir_md_hn'])
                room_id.append(s['room_id_hn'])
            if 'reverebed_audio_hp' in s and s['reverebed_audio_hp'] is not None:
                reverbed_audios.append(s['reverebed_audio_hp'])
                audios_idx.append(s['audio_hp_idx'])
                rirs_idx.append(s['rir_idx'])
                rirs.append(s['RIR'])
                rirs_md.append(s['rir_md'])
                room_id.append(s['room_id'])
            for k in extra_kwargs:
                if k in s:
                    extra_kwargs_flatten[k].append(s[k])
                    if 'reverebed_audio_hn' in s and s['reverebed_audio_hn'] is not None:
                        extra_kwargs_flatten[k].append(s[k])  # add it again
                    if 'reverebed_audio_hp' in s and s['reverebed_audio_hp'] is not None:
                        extra_kwargs_flatten[k].append(s[k]) # add it again

        # handle reverbed audio
        max_speech = np.max([x.size for x in reverbed_audios])
        reverb_speech_samples_batch = [pad_array(x, max_speech) for x in reverbed_audios]
        reverb_speech_samples_batch = np.vstack(reverb_speech_samples_batch)

        # handle rirs
        max_rir = nsample_rir
        rir_samples_batch = [pad_array(x, max_rir) for x in rirs]
        rir_samples_batch = np.vstack(rir_samples_batch)

        # calc similarity matrix for loss
        sim_mat_gt = calc_sim_mat_for_loss(batch_size=len(rir_samples_batch),
                                           rirs_idx=rirs_idx,
                                           audios_idx=audios_idx,
                                           verbose=verbose,
                                           return_tensors='pt')
        processor_out = {}
        if processor is not None:
            processor_out = processor(rtfs=rir_samples_batch,
                                      audios=reverb_speech_samples_batch,
                                      sampling_rate=fs,
                                      return_tensors='pt')

        processor_out['return_loss'] = True
        processor_out['return_dict'] = True
        processor_out['audios'] = reverb_speech_samples_batch
        processor_out['sim_mat_gt'] = sim_mat_gt
        processor_out['rir_idx'] = rirs_idx
        processor_out['audios_idx'] = audios_idx
        processor_out['rir_md'] = rirs_md

        for k in extra_kwargs:
            if extra_kwargs_flatten[k]:
                assert k not in processor_out, f"{k} already in {processor_out.keys()}"
                processor_out[k] = extra_kwargs_flatten[k]

        return processor_out

    return inner_function


def example_run_dataset():
    from ..utils import get_audio_paths, play_audio_ffplay
    training_dataset = 'libri-dev-other'
    n_samples_training = 200
    fs = 8000
    audio_paths = get_audio_paths(training_dataset)[:n_samples_training]
    print(f'selected {n_samples_training} audio samples for training, in practice we got {len(audio_paths)}')

    from ..utils import get_generater_rirs_paths
    rir_paths, md_paths = get_generater_rirs_paths()

    dataset = CARIR_dataset_V3(audio_paths=audio_paths,
                               rir_paths=rir_paths,
                               md_paths=md_paths,
                               fs=fs,                     # Hz
                               hard_negative_prob=1.0,
                               hard_positive_prob=1.0,
                               )
    if True:
        batch = []
        for i in np.random.permutation(len(dataset))[:100]:
            batch.append(dataset[i])
        output = collate_fn_v3(batch, nsample_rir=dataset.max_rir_length, verbose=True)
    else:
        for i in np.random.permutation(len(dataset)):
            output = dataset[i]
            # from IPython import embed; embed()
            if True:
                play_audio_ffplay(10 * output['reverbed_audio1'], dataset.fs)
                play_audio_ffplay(10 * output['reverebed_audio_hn'], dataset.fs)
                play_audio_ffplay(10 * output['reverebed_audio_hp'], dataset.fs)
                plt.subplot(311)
                plt.plot(output['audio1'])
                plt.ylabel('orig')
                plt.subplot(312)
                plt.plot(output['reverbed_audio1'])
                plt.ylabel('reverebed')
                plt.subplot(313)
                plt.plot(output['RIR'])
                plt.ylabel('rir')
                plt.show()
            print(output['rir_md'])

            # d = {'l': [],
            #      'e': [],
            #      'i': [],}
            # for j, rirs in enumerate(dataset.rirs):
            #     for r_ in rirs:
            #         d['l'].append(len(r_))
            #         d['e'].append(np.sum(r_ ** 2))
            #         d['i'].append(j)
            # df = pd.DataFrame(d)
            # for l, df_ in df.groupby('i'):
            #     plt.scatter(df_['l'], df_['e'], label=os.path.basename(dataset.rir_paths[l]).split('_')[1].split('.')[0])
            # plt.xlabel('RIR length')
            # plt.ylabel('RIR energy')
            # plt.legend()
            # plt.show()


if __name__ == '__main__':
    example_run_dataset()