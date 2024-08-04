import os
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from ..utils import pad_array
from .enc_dec_dataset import CARIR_AutoEncoder_dataset


class roomClassifierDataset(CARIR_AutoEncoder_dataset):
    def __init__(self, is_v4 = False, skip_audio=True, *args, **kwargs):
        super(roomClassifierDataset, self).__init__(skip_audio=skip_audio, *args, **kwargs)
        self.num_classes = len(self.rir_paths)
        self.is_v4 = is_v4
        if self.is_v4:
            self.worker_init_function(0)  # init self.rooms
            self.num_classes = len(self.rooms)

    def __getitem__(self, indx_):
        data_out = super(roomClassifierDataset, self).__getitem__(indx_)
        if self.is_v4:
            data_out["gt"] = data_out["v4_room_indx"]
        else:
            assert 0
            data_out["gt"] = data_out["room_id"]
        return data_out


class roomClassifierMCDataset(roomClassifierDataset):
    def __init__(self,
                 return_room_dim_as_is,
                 w_min_length = None,
                 w_stride = None,
                 *args, **kwargs):
        super(roomClassifierMCDataset, self).__init__(*args, **kwargs)
        self.return_room_dim_as_is = return_room_dim_as_is
        self.w_min_length = w_min_length
        self.w_stride = w_stride

    def __len__(self):
        return self.total_rirs_num

    def __getitem__(self, indx_):
        data_out = super(roomClassifierDataset, self).__getitem__(indx_)
        rir_md = data_out["rir_md"]
        try:
            L_new = eval(', '.join(rir_md["L_new"].split()))
        except:
            L_new = eval(', '.join(rir_md["L_new"].replace('[', '').replace(']', '').split()))
        if self.return_room_dim_as_is:
            data_out["gt"] = L_new
        else:
            data_out["gt"] = [int((wall_l - self.w_min_length[i]) / self.w_stride[i]) for i, wall_l in enumerate(L_new)]
        return data_out


def collate_fn(feature_extractor, audio_feature_extractor=None, nsample_rir=3200):
    def inner_function(batch):
        if  "reverbed_audio1" in batch[0]:
            speech_size = [x['reverbed_audio1'].size for x in batch]
            max_speech = np.max(speech_size)
            reverb_speech_samples_batch = [pad_array(x['reverbed_audio1'], max_speech) for x in batch]
            reverb_speech_samples_batch = np.vstack(reverb_speech_samples_batch)
            processor_out = {}
            processor_out["input_audio_features"] = audio_feature_extractor(reverb_speech_samples_batch, return_tensors='pt')["input_features"]
        else:
            rir_samples_batch = [pad_array(x['RIR'], nsample_rir) for x in batch]
            rir_samples_batch = np.vstack(rir_samples_batch)
            processor_out = feature_extractor(rirs=rir_samples_batch, return_tensors='pt')
        processor_out["gt"] = torch.tensor(np.array([x["gt"] for x in batch]))
        if processor_out["gt"].dtype == torch.float64:
            processor_out["gt"] = processor_out["gt"].float()

        #"speaker_id", "audio1_fpath", "audio_rir_corr", "RIR"
        for k in ["rir_md", "rir_idx", "room_id", "v4_room_indx"]:
            if k in batch[0]:
                assert k not in processor_out, f"{k} already in {processor_out.keys()}"
                processor_out[k] = [x[k] for x in batch]
        return processor_out
    return inner_function


def example_run_dataset():
    from ..utils import get_generater_rirs_paths
    rir_paths, md_paths = get_generater_rirs_paths()
    dataset = roomClassifierDataset(rir_paths=rir_paths,
                                    md_paths=md_paths,
                                    )
    for i in np.random.permutation(len(dataset)):
        output = dataset[i]
        if True:
            plt.plot(output['RIR'])
            plt.show()
        print(output['rir_md'])
        from IPython import embed; embed()


if __name__ == '__main__':
    example_run_dataset()