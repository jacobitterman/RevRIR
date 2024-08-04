import os
import json
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..utils import get_generater_rirs_paths, makedirs_p
from ..data.enc_dec_dataset import CARIR_AutoEncoder_dataset, collate_fn
from ..feature_extraction_carir import RtfFeatureExtractor
from ..models.enc_dec_rtf import EncoderDecoder, RtfModelConfig



def eval(model_path, config_path = None):
    rir_paths, md_paths = get_generater_rirs_paths("test")
    dataset = CARIR_AutoEncoder_dataset(rir_paths=rir_paths,
                                        md_paths=md_paths,
                                        )

    if config_path is None:
        config_path =  os.path.join(os.path.dirname(model_path), "config.conf")
    config = RtfModelConfig.from_dict(json.load(open(config_path, "r")))

    model = EncoderDecoder(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    feature_extractor = RtfFeatureExtractor(feature_size = -1, sampling_rate=8000, padding_value = 0)
    nsample_rir = (config.in_dim - 1) * 2
    collate_fn_p = collate_fn(feature_extractor=feature_extractor,
                              nsample_rir=nsample_rir,
                              )
    # from ..data.dummy_dataset import DummyDataset
    # dataset = DummyDataset(dataset, 50)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers = 0,
                            batch_size = 50,
                            collate_fn=collate_fn_p,
                            drop_last=True,
                            worker_init_fn=dataset.worker_init_function)

    dataloader_iter = iter(dataloader)
    rir_b = next(dataloader_iter)
    with torch.no_grad():
        model_input = rir_b["input_rtf_features"]
        rtf_fft = model(model_input)

        for i in range(model_input.shape[0]):
            ax = plt.subplot(311)
            plt.plot(rtf_fft["rtf_features"][i])
            plt.subplot(312, sharex=ax, sharey=ax)
            plt.plot(model_input[i])
            plt.subplot(313, sharex=ax, sharey=ax)
            plt.plot(rtf_fft["rtf_features"][i])
            plt.plot(model_input[i])
            plt.show()


import fire
if __name__ == '__main__':
    fire.Fire()