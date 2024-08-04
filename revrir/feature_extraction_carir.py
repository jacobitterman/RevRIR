# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Feature extractor class for CLAP."""


from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import copy
import librosa


import torchaudio.compliance.kaldi as ta_kaldi

from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

class CarirFeatureExtractor(SequenceFeatureExtractor):
    def __init__(
        self,
        feature_size=64,
        sampling_rate=48_000,
        hop_length=480,
        max_length_s=10,
        fft_window_size=1024,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        frequency_min: float = 0,
        frequency_max: float = 14_000,
        top_db: int = None,
        truncation: str = "fusion",
        padding: str = "repeatpad",
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.fft_window_size = fft_window_size
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate


    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance, excpet for the
            mel filter banks, which do not need to be saved or printed as they are too long.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "mel_filters_slaney" in output:
            del output["mel_filters_slaney"]
        return output

    def torch_imp(self,
                  raw_speech: Union[np.ndarray, List[np.ndarray]],
                  sampling_rate):
        fbank = [ta_kaldi.fbank(
            torch.Tensor(audio[None, :]),
            htk_compat=True,
            sample_frequency=sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.feature_size,
            dither=0.0,
            frame_length=self.n_fft / sampling_rate * 1000,
            frame_shift=self.hop_length / sampling_rate * 1000,  # in ms
            snip_edges=False)
            for audio in raw_speech]
        input_mel = torch.stack(fbank)[:, None]
        return input_mel

    def librosa_imp(self,
                    raw_speech: Union[np.ndarray, List[np.ndarray]],
                    sampling_rate=None,
                    ):
        out = [librosa.power_to_db(librosa.feature.melspectrogram(y=a,
                                              sr=sampling_rate,
                                              n_fft=self.n_fft,
                                              hop_length=self.hop_length,
                                              win_length=self.fft_window_size,
                                              n_mels=self.feature_size)) for a in raw_speech]
        input_mel = np.array(out, dtype=np.float32)[:, np.newaxis].transpose(0, 1, 3, 2)
        return input_mel

    def __call__(
            self,
            raw_speech: Union[np.ndarray, List[np.ndarray]],
            sampling_rate = None,
            return_tensors: Optional[str] = None,
            **kwargs,
    ) -> BatchFeature:

        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        input_mel = self.torch_imp(raw_speech, sampling_rate)  # should be faster
        # input_mel = self.librosa_imp(raw_speech, sampling_rate)

        input_features = {"input_features": input_mel, "is_longer": [False] * input_mel.shape[0]}
        input_features = BatchFeature(input_features)

        if return_tensors is not None:
            input_features = input_features.convert_to_tensors(return_tensors)

        return input_features


class RtfFeatureExtractor(SequenceFeatureExtractor):
    def __init__(self,
                 use_phase: bool = False,
                 remove_dc_offset: bool = True,
                 process_time_domain: bool = False,
                 **kwargs):
        super().__init__(
            **kwargs,
        )
        self.use_phase = use_phase
        self.remove_dc_offset = remove_dc_offset
        self.process_time_domain = process_time_domain

    def __call__(self,
                 rirs,
                 return_tensors='pt',
                 ):
        if not isinstance(rirs, list) and len(rirs.shape) == 1:
            rirs = [rirs]
        if not isinstance(rirs, np.ndarray):
            rirs = np.array(rirs, dtype=np.float32)
        assert len(rirs.shape) == 2

        outputs = []
        for rir in rirs:
            if self.remove_dc_offset:
                rir = rir - rir.mean()

            if self.process_time_domain:
                output = rir
            else:
                rtf = np.fft.rfft(rir)
                mag = 20 * np.log10(np.abs(rtf) + 1e-10)

                if self.use_phase:
                    phase = np.angle(rtf)
                    output = np.vstack((mag, phase)).astype(np.float32)
                else:
                    output = mag.astype(np.float32)
            outputs.append(output)

        outputs = np.array(outputs).astype(np.float32)
        if return_tensors == 'pt':
            import torch
            outputs = torch.tensor(outputs, dtype=torch.float32)
        elif return_tensors == 'np':
            pass
        else:
            raise ValueError(f'unsupported return_tensor type: {return_tensors}')

        return {'input_rtf_features': outputs}
