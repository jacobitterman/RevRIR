import os
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from ..utils import pad_array
from ..eval.classifier_benchmark import name2id

from .dataset_v2 import CARIR_dataset


class CARIR_AutoEncoder_dataset(CARIR_dataset):
    def __init__(self,
                 rir_paths,
                 md_paths,
                 mode: str = "train",
                 skip_audio = True,
                 audio_paths=None,
                 **kwargs
                 ):
        super(CARIR_AutoEncoder_dataset, self).__init__(audio_paths, rir_paths, md_paths, mode=mode, skip_audio = skip_audio, **kwargs)
        self.rirs_md = [pd.read_csv(md_p) for md_p in self.md_paths]
        self.len = sum([len(x) for x in self.rirs_md])


def collate_fn(feature_extractor, nsample_rir=3200):
    def inner_function(batch):
        rir_samples_batch = [pad_array(x['RIR'], nsample_rir) for x in batch]
        rir_samples_batch = np.vstack(rir_samples_batch)
        processor_out = feature_extractor(rirs=rir_samples_batch, return_tensors='pt')

        for k in ["rir_md", "rir_idx", "v4_room_indx"]:
            if k in batch[0]:
                assert k not in processor_out, f"{k} already in {processor_out.keys()}"
                processor_out[k] = [x[k] for x in batch]
        return processor_out
    return inner_function


def example_run_dataset():
    from ..utils import get_generater_rirs_paths
    rir_paths, md_paths = get_generater_rirs_paths()
    dataset = CARIR_AutoEncoder_dataset(rir_paths=rir_paths,
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