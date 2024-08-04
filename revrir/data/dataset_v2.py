import os
import torch
import random

import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import scipy.signal as ss
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from .data_utils import get_sim_mat_and_min_exp_loss
from ..utils import get_audio, pad_array, rms
from ..eval.classifier_benchmark import name2id
from ..g729a_utils import G729A


v4_versions = ["v4", "v4.1"]
v5_versions = ["v5"]
v_gt_4_versions = v4_versions + v5_versions

class CARIR_dataset(Dataset):
    def __init__(self,
                 audio_paths,
                 rir_paths,
                 md_paths,
                 fs=8000, # Hz
                 max_audio_length_sec=10,
                 mode: str = "train",
                 reverb = True,
                 rms_augmentation_prob = -1,
                 rms_factor_augmentation_range = (2, 2.5),
                 skip_audio = False,
                 compress_audio_prob=-1,
                 sample_rooms_by_bins = True,
                 skip_rir_trad_feat=False,
                 ):

        self.mode = mode
        self.reverb = reverb

        # rirs
        if isinstance(rir_paths, str):
            rir_paths = [rir_paths]
        self.rirs = None  # in worker_init_function
        self.rirs_traditional_features = None # in worker_init_function
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
        self.md_paths = md_paths
        self.rirs_md = None  # in worker_init_function

        self.n_samples = None  # in worker_init_function
        self.n_samples_zero = None  # in worker_init_function
        self.max_rir_length = None  # in worker_init_function
        self.rir_iter = None  # in worker_init_function
        self.total_rirs_num = None  # in worker_init_function
        self.rir_data_version = None  # in worker_init_function
        self.rooms = None   # in worker_init_function. relevant for v4 data
        self.n_samples_per_room = None  # in worker_init_function. relevant for v4 data
        self.sample_rooms_by_bins = sample_rooms_by_bins
        self.skip_rir_trad_feat = skip_rir_trad_feat

        # audios
        self.fs = fs
        self.max_audio_length_sec = max_audio_length_sec
        self.audio_paths = audio_paths
        self.len = len(self.audio_paths) if self.audio_paths is not None else -1
        self.audios = None  # in worker_init_function

        self.rms_augmentation_prob = rms_augmentation_prob
        self.rms_augmentation_range = rms_factor_augmentation_range
        self.augment_audio = rms_augmentation_prob > 0.0


        self.skip_audio = skip_audio
        self.compressor = G729A()
        self.compress_audio_prob = compress_audio_prob

        print('finished __init__ CARIR dataset')

    def init_rir_stuff(self, worker_index):
        if worker_index == 0:
            print(f"loading {len(self.rir_paths)} rir files")
        self.rirs_data = [pkl.load(open(r_p, 'rb')) for r_p in self.rir_paths]
        self.rirs = [r[-1] for r in self.rirs_data]
        self.rirs_md = [pd.read_csv(md_p) for md_p in self.md_paths]
        max_rir_length_per_file = None
        if len(self.rirs_data[0]) >= 3:  # each file contain: version, max_rir, ..., rirs
            self.rir_data_version = self.rirs_data[0][0]
            print(f"reading rir files version {self.rir_data_version}")
            max_rir_length_per_file = [r[1] for r in self.rirs_data]
            if len(self.rirs_data[0]) >= 5:  # each file contain: version, max_rir, rooms, n_samples_per_room, [rirs_traditional_features] ,rirs
                assert len(self.rirs_data[0]) > 5 or self.rir_data_version in v4_versions, f"expected data from v4 but got {self.rir_data_version}"
                self.rooms = [x for r in self.rirs_data for x in r[2]]  # flatten

                n_samples_per_room = [r[3] for r in self.rirs_data]
                self.n_samples_per_room = [[x] * len(self.rirs_data[i][2]) for i, x in enumerate(n_samples_per_room)]
                self.n_samples_per_room = [x for r in self.n_samples_per_room for x in r]  # flatten
            if len(self.rirs_data[0]) == 6:
                assert self.rir_data_version in v5_versions, f"expected data from v5 but got {self.rir_data_version}"
                self.rirs_traditional_features = [r[-2] for r in self.rirs_data]
                for x, y in zip(self.rirs_traditional_features, self.rirs):
                    assert len(x) == len(y)
        else:
            max_rir_length_per_file = [r.shape[-1] for r in self.rirs_data]

        self.total_rirs_num = sum([len(r) for r in self.rirs])

        self.n_samples = np.cumsum([len(x) for x in self.rirs])
        self.n_samples_zero = np.hstack((0, self.n_samples[:-1]))
        self.max_rir_length = np.max(max_rir_length_per_file)
        self.rir_iter = self.rir_itr_fn()

    def init_audio_stuff(self, worker_index):
        if self.skip_audio:
            self.len = -1
        else:
            # TODO: use seed per worker_index?
            if self.audio_paths[0].endswith(('.ibin', '.bin')):
                from ..utils import MultiBinary, decode_buff
                from functools import partial
                self.audios = MultiBinary(self.audio_paths, getitem_fn=partial(decode_buff, decode_audio_dtype=np.float32))
            else:
                self.audios = [get_audio(p, self.fs)[:self.max_audio_length_sec * self.fs] for p in
                               tqdm(self.audio_paths, desc=f'loading audios in worker {worker_index}')]

            self.len = len(self.audios)

    def worker_init_function(self, worker_index):
        self.init_audio_stuff(worker_index)
        self.init_rir_stuff(worker_index)
        if self.skip_audio:
            self.len = self.total_rirs_num

    def get_rir(self, idx):
        room_file_id = np.where(idx < self.n_samples)[0][0]  # 0, 1, 2
        idx_within_room = idx - self.n_samples_zero[room_file_id]
        rir = self.rirs[room_file_id][idx_within_room]
        rir_md = self.rirs_md[room_file_id].iloc[idx_within_room]
        v4_room_indx = -1 if self.rir_data_version not in v_gt_4_versions else self.rooms.index(eval(rir_md['L']))
        rir_trad_feats = None
        if self.rirs_traditional_features is not None:
            rir_trad_feats = self.rirs_traditional_features[room_file_id][idx_within_room]
        return rir, rir_trad_feats, rir_md, room_file_id, idx_within_room, v4_room_indx

    def rir_itr_fn(self):
        rir_epoch = 0
        while True:
            print(f'starting epoch #{rir_epoch} in mode {self.mode}')
            if self.rir_data_version in v_gt_4_versions and self.sample_rooms_by_bins:
                ind_iter = binned_permutation(self.n_samples_per_room)
            else:
                ind_iter = np.random.permutation(self.n_samples[-1])
            for i in ind_iter:
                rir, rir_trad_feats, rir_md, room_id, _, v4_room_indx = self.get_rir(i)
                yield rir, rir_trad_feats,rir_md, i, self.rir_path2id[room_id], v4_room_indx
            rir_epoch += 1

    def __len__(self):
        if self.mode == "train":
            return max(self.len, self.total_rirs_num)
        return self.len

    def __getitem__(self, indx_):
        if self.rirs is None:
            self.worker_init_function(0)

        # retrieve rir
        rir, rir_trad_feat, rir_md, rir_id, room_id, v4_room_indx = next(self.rir_iter)
        # print(f"rms is: {rms(audio)}")

        if not self.skip_audio:
            # retrieve audios and audios_path
            indx = indx_ % self.len
            if isinstance(self.audios, list):
                audio = self.audios[indx][: self.max_audio_length_sec * self.fs]
                audio_path = self.audio_paths[indx]
            else:
                bin_sample = self.audios[indx]
                audio = bin_sample['audio'][: self.max_audio_length_sec * self.fs]
                audio_path = bin_sample['audio_path']

            if self.reverb:
                reverebed_audio = ss.convolve(rir, audio)
            else:
                reverebed_audio = audio

                # get room id from audio_path if there's no RIR:
                audio_basename = os.path.basename(audio_path)
                room_id = None
                for k in name2id:
                    if k in audio_basename:
                        room_id = name2id[k]
                assert room_id is not None, audio_basename

            reverebed_audio = reverebed_audio[: self.max_audio_length_sec * self.fs]
            assert len(reverebed_audio) <= self.max_audio_length_sec * self.fs

            # augmentations if needed:
            if self.augment_audio:
                reverebed_audio = augment_rms(reverebed_audio, self.rms_augmentation_prob, self.rms_augmentation_range)

            if self.compress_audio_prob > 0 and np.random.random() < self.compress_audio_prob:
                reverebed_audio = self.compressor.apply(reverebed_audio)

            speaker_id = audio_path.rsplit('/',4)[-3]
            book_id = audio_path.rsplit('/',4)[-2]

        data_out = {
            'rir_idx': rir_id,
            'RIR': rir,
            'RIR_trad_feat': None if self.skip_rir_trad_feat else rir_trad_feat,
            'rir_md': rir_md.to_dict(),
            'room_id': room_id,
            'v4_room_indx': v4_room_indx,
        }
        if not self.skip_audio:
            data_out.update({
                'audio1': audio,
                'audio1_idx': indx,
                'audio1_fpath': audio_path,
                'speaker_id': speaker_id,
                'reverbed_audio1': reverebed_audio,
                'audio_rir_corr': f'{speaker_id}_{book_id}_{rir_id}',
            })
        return data_out


def collate_fn(processor, fs, nsample_rir=3200, save_rms=True, name_conversion_map=None):
    def inner_function(batch):
        # for x in batch:
        #     x['reverbed_audio1'] = x['reverbed_audio1'] if 'real_recording' in x['audio1_fpath'] else x['reverbed_audio1']  # TODO: add param to normalize audio?

        if save_rms:
            speech_rms = [rms(x['reverbed_audio1']) for x in batch]

        speech_size = [x['reverbed_audio1'].size for x in batch]
        max_speech = np.max(speech_size)
        reverb_speech_samples_batch = [pad_array(x['reverbed_audio1'], max_speech) for x in batch]
        reverb_speech_samples_batch = np.vstack(reverb_speech_samples_batch)

        max_rir = nsample_rir
        rir_samples_batch = [pad_array(x['RIR'], max_rir) for x in batch]
        rir_samples_batch = np.vstack(rir_samples_batch)
        processor_out = processor(rtfs=rir_samples_batch,
                                  audios=reverb_speech_samples_batch,
                                  sampling_rate=fs,
                                  return_tensors='pt')
        if batch[0]['RIR_trad_feat'] is not None:
            trad_feats = np.vstack([x['RIR_trad_feat'] for x in batch])
            assert len(trad_feats) == len(rir_samples_batch)
            stacked_feats = torch.hstack((torch.tensor(trad_feats, dtype=torch.float32),
                                          processor_out['input_rtf_features']))
            processor_out['input_rtf_features'] = stacked_feats

        processor_out['return_loss'] = True
        processor_out['return_dict'] = True
        processor_out['audios'] = reverb_speech_samples_batch
        processor_out['speech_size'] = speech_size
        if save_rms:
            processor_out['speech_rms'] = speech_rms

        for k in ["speaker_id", "audio1_fpath", "rir_md", "rir_idx", "audio_rir_corr", "RIR", "room_id", "v4_room_indx", "RIR_trad_feat"]:
            if k in batch[0]:
                assert k not in processor_out, f"{k} already in {processor_out.keys()}"
                processor_out[k] = [x[k] for x in batch]

        if batch[0]["v4_room_indx"] >= 0:
            labels, min_loss = get_sim_mat_and_min_exp_loss(processor_out["v4_room_indx"])

            processor_out["labels"] = torch.Tensor(labels)
            processor_out['min_expected_loss'] = min_loss

        if name_conversion_map:
            for k, v in name_conversion_map.items():
                if k in processor_out:
                    processor_out[v] = processor_out[k]
                    del processor_out[k]

        return processor_out
    return inner_function


def augment_rms(audio, rms_augmentation_prob, rms_augmentation_range):
    min_, max_ = rms_augmentation_range
    if np.random.random() <= rms_augmentation_prob:
        rms_factor = np.random.random() * (max_ - min_) + min_  # TODO: check correctness
        audio = audio * rms_factor
    return audio


def binned_permutation(bin_sizes):
    # for the V4 data

    # Create a list of indices for each bin
    bin_indices = [list(range(sum(bin_sizes[:i]), sum(bin_sizes[:i+1]))) for i in range(len(bin_sizes))]

    # Shuffle each list of indices independently
    shuffled_indices = [random.sample(indices, len(indices)) for indices in bin_indices]

    # Flatten the list of shuffled indices
    #flattened_indices = [index for sublist in shuffled_indices for index in sublist]
    flattened_indices = [index for sublist in zip(*shuffled_indices) for index in sublist]

    # Determine the maximum bin size
    max_bin_size = max(bin_sizes)

    # Pad the shuffled indices with None for bins with fewer indices
    padded_indices = [indices + [None] * (max_bin_size - len(indices)) for indices in shuffled_indices]

    # Interleave the shuffled indices, removing padding
    flattened_indices = [index for sublist in zip(*padded_indices) for index in sublist if index is not None]

    # shuffle indices of each bins epoch [ = len(bin_sizes) elements]
    for i in range(0, len(flattened_indices), len(bin_sizes)):
        max_i = min(len(flattened_indices), i+len(bin_sizes))
        flattened_indices[i:max_i] = np.random.permutation(flattened_indices[i:max_i])

    return flattened_indices


def example_run_dataset():
    from ..utils import get_audio_paths, play_audio_ffplay
    from torch.utils.data import DataLoader
    training_dataset = 'libri-dev-other'
    n_samples_training = 200
    fs = 8000
    audio_paths = get_audio_paths(training_dataset)[:n_samples_training]
    print(f'selected {n_samples_training} audio samples for training, in practice we got {len(audio_paths)}')

    from ..utils import get_generater_rirs_paths
    rir_paths, md_paths = get_generater_rirs_paths("v5_benchmark_3k")
    dataset = CARIR_dataset(audio_paths=audio_paths,
                            rir_paths=rir_paths,
                            md_paths=md_paths,
                            fs=fs,                     # Hz
                            )
    output = dataset[1]
    from functools import partial
    from ..processing_carir import CarirProcessor
    pretrained_dir = '/home/talr/Downloads/CARIR/CARIR_model/open_slr_68k_ckpt_49/open_slr_68k_ckpt_49_2/preprocessor_config.json'
    carir_processor = CarirProcessor.from_pretrained(pretrained_dir)
    collate_fn_p = collate_fn(processor=carir_processor,
                              fs=8000,
                              nsample_rir=5000,
                              )

    data_loader = DataLoader(dataset, batch_size=3, shuffle=True, collate_fn=collate_fn_p)

    a = [dataset[i]["v4_room_indx"] for i in range(300)]
    for i in np.random.permutation(len(dataset)):
        output = dataset[i]
        if False:
            play_audio_ffplay(output['reverbed_audio1'], dataset.fs)
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
        from IPython import embed; embed()


if __name__ == '__main__':
    example_run_dataset()
