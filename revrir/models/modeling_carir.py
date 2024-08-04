# coding=utf-8
# Copyright 2023 The LAION-AI Team and The HuggingFace Team. All rights reserved.
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
""" PyTorch Carir model."""
import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sklearn.metrics import roc_auc_score

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.models.clap.modeling_clap import ClapAudioModelOutput, ClapAudioModel, \
    ClapProjectionLayer

logger = logging.get_logger(__name__)

CarirAudioModelOutput = ClapAudioModelOutput
CarirProjectionLayer = ClapProjectionLayer
CarirAudioModel = ClapAudioModel

from ..configuration_carir import CarirAudioConfig, CarirConfig, CarirRtfConfig
from ..loss import hinge_loss


def contrastive_loss(logits: torch.Tensor, labels = None) -> torch.Tensor:
    if labels is None:
        labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Clap
class CarirRTFModelOutput(ModelOutput):
    """
    Base class for text model's outputs that also contains a pooling of the last hidden states.

    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    rtf_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPOutput with CLIP->Clap, vision->audio, Vision->Audio, image->audio
class CarirOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits_per_audio: Optional[torch.Tensor] = None
    logits_per_rtf: Optional[torch.Tensor] = None
    rtf_embeds: torch.FloatTensor = None
    audio_embeds: torch.FloatTensor = None
    rtf_model_output: BaseModelOutputWithPooling = None
    audio_model_output: BaseModelOutputWithPooling = None
    cosine_sim: Optional[torch.Tensor] = None
    auc_score: float = None

    def to_tuple(self) -> Tuple:
        return tuple(
            self[k] if k not in ["rtf_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )


class CarirPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CarirConfig
    base_model_prefix = "carir"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor

        if isinstance(module, CarirModel):
            nn.init.normal_(module.logit_scale_a, std=factor * 0.02)
            nn.init.normal_(module.logit_scale_r, std=factor * 0.02)

    def _set_gradient_checkpointing(self, module, value=False):
        pass

class RtfModel(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.config = config
        in_dim = config.in_dim
        out_dim = config.out_dim

        num_mid_layers = config.num_mid_layers
        print(f"setting in_dim of RtfModel from {in_dim} to {in_dim + config.classic_feature_dim}")
        in_dim = in_dim + config.classic_feature_dim

        if hasattr(config, "old_mid_layers"):
            dim_delta = in_dim // 2 + 1   # backword compatibility
        else:
            dim_delta = int((in_dim - out_dim) / (num_mid_layers + 1))  # positive value for encoder and negative for encoder

        dims = [in_dim]
        for i in range(num_mid_layers):
            mid_dim = dims[-1] - dim_delta
            dims.append(mid_dim)
        dims.append(out_dim)

        if config.use_bn:
            self.linears = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.BatchNorm1d(dims[i+1]))
                                          for i in range(num_mid_layers)])
        else:
            self.linears = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
                                          for i in range(num_mid_layers)])
        self.last_linear = nn.Linear(dims[-2], out_dim)
        self.out_dim = out_dim

    def forward(self, rtf_inputs, return_dict=False):
        # ModuleList can act as an iterable, or be indexed using ints
        x = rtf_inputs
        for i, l in enumerate(self.linears):
            x = l(x)
        out =  self.last_linear(x)
        if return_dict:
            return {'rtf_features': out, 'last_hidden_state': x}
        return out, x

    def fix_state_dict(self, state_dict, prefix):
        dict_keys = list(state_dict.keys())
        prefix_len = len(prefix)
        for k in dict_keys:
            new_name = None
            if k.startswith(prefix + 'linear1'):
                new_name = k.replace('linear1', 'linears.0.0')
            elif k.startswith(prefix + 'linear2'):
                new_name = k.replace('linear2', 'last_linear')
            else:
                continue

            if new_name is not None:
                state_dict[new_name] = state_dict[k]
                del state_dict[k]
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        """ Overrides state_dict() to load values from previous format of StackedResnet"""
        state_dict = self.fix_state_dict(state_dict, "")
        super().load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """ Overrides _load_from_state_dict() to load values from previous format of StackedResnet"""
        state_dict = self.fix_state_dict(state_dict, prefix)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class CarirRtfModel(CarirPreTrainedModel):
    config_class = CarirRtfConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.model = RtfModel(self.config)
        self.dim = self.model.out_dim

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, CarirRTFModelOutput]:
        model_output = self.model(input_features, True)

        rtf_embeds = model_output['rtf_features']

        return rtf_embeds

class CarirModel(CarirPreTrainedModel):
    config_class = CarirConfig

    def __init__(self, config: CarirConfig):
        super().__init__(config)

        if not isinstance(config.rtf_config, CarirRtfConfig):
            raise ValueError(
                "config.rtf_config is expected to be of type CarirRtfConfig but is of type"
                f" {type(config.rtf_config)}."
            )

        if not isinstance(config.audio_config, CarirAudioConfig):
            raise ValueError(
                "config.audio_config is expected to be of type CarirAudioConfig but is of type"
                f" {type(config.audio_config)}."
            )

        self.rtf_config = config.rtf_config
        self.audio_config = config.audio_config

        self.logit_scale_a = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.logit_scale_r = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))

        self.projection_dim = config.projection_dim

        self.rtf_model = CarirRtfModel(self.rtf_config)
        self.rtf_projection = CarirProjectionLayer(self.rtf_config)

        self.audio_model = CarirAudioModel(self.audio_config)
        self.audio_projection = CarirProjectionLayer(self.audio_config)

        self.calc_auc = True

        self.orig_loss = self.config.use_orig_loss
        self.margin = self.config.hinge_loss_margin
        if not self.orig_loss:
            self.calc_auc = False

        self.dual_loss = self.config.dual_loss

        # Initialize weights and apply final processing
        self.post_init()

    def get_embedder1_latents(self, *args, **kwargs):
        return self.get_audio_features(*args, **kwargs)

    def get_embedder2_latents(self, *args, **kwargs):
        return self.get_rtf_features(*args, **kwargs)

    def get_rtf_features(
        self,
        input_rtf_features: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        rtf_embeds = self.rtf_model(
            input_features=input_rtf_features,
            return_dict=return_dict,
        )
        rtf_features = self.rtf_projection(rtf_embeds)

        rtf_features = torch.nn.functional.normalize(rtf_features, dim=-1)

        return rtf_features

    def get_audio_features(
        self,
        input_audio_features: Optional[torch.Tensor] = None,
        is_longer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_audio_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        audio_features = self.audio_projection(pooled_output)
        audio_features = torch.nn.functional.normalize(audio_features, dim=-1)

        return audio_features

    def forward(
        self,
        input_rtf_features: Optional[torch.LongTensor] = None,
        input_audio_features: Optional[torch.FloatTensor] = None,
        labels = None,
        is_longer: Optional[torch.BoolTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CarirOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        audio_outputs = self.audio_model(
            input_features=input_audio_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        rtf_embeds = self.rtf_model(
            input_features=input_rtf_features,
            return_dict=return_dict,
        )

        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)

        rtf_embeds = self.rtf_projection(rtf_embeds)

        # normalized features
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        rtf_embeds = rtf_embeds / rtf_embeds.norm(p=2, dim=-1, keepdim=True)

        loss = None

        if self.orig_loss:
            # cosine similarity as logits
            logits_per_rtf = torch.matmul(rtf_embeds, audio_embeds.t()) * self.logit_scale_r.exp()
            logits_per_audio = torch.matmul(audio_embeds, rtf_embeds.t()) * self.logit_scale_a.exp()
            if return_loss:
                rtf_loss = contrastive_loss(logits_per_rtf, labels)
                audio_loss = contrastive_loss(logits_per_audio.t(), labels)
                if self.dual_loss and labels is not None:
                    logits_rtf_rtf = torch.matmul(rtf_embeds, rtf_embeds.t())
                    logits_audio_audio = torch.matmul(audio_embeds, audio_embeds.t())
                    rtf_rtf_loss = contrastive_loss(logits_rtf_rtf, labels)
                    audio_audio_loss = contrastive_loss(logits_audio_audio, labels)
                    print(f"{rtf_rtf_loss=}, {audio_audio_loss=}")
                    loss = (rtf_rtf_loss + audio_audio_loss + rtf_loss + audio_loss) / 4.0
                else:
                    loss = (rtf_loss + audio_loss) / 2.0

        else:
            logits_per_rtf = None
            logits_per_audio = None
            if return_loss:
                similarity_matrix = kwargs["sim_mat_gt"]
                loss, pos_loss, neg_loss, mean_neg_loss, mean_pos_loss = hinge_loss(similarity_matrix, audio_embeds, rtf_embeds,
                                                      margin=self.margin, alpha=0.48,
                                                      apply_softmax_on_neg = True,
                                                      temperature = 0.01)  # TODO add to config
                print(f"loss is: {loss}, pos_loss: {pos_loss}, neg_loss: {neg_loss}, mean_neg_loss: {mean_neg_loss}, mean_pos_loss: {mean_pos_loss}")

        auc_score = -1.
        cosine_sim = None
        if self.calc_auc:
            cosine_sim = torch.einsum('i d, j d -> i j', audio_embeds, rtf_embeds)
            auc_score = self.get_auc(cosine_sim)
        if loss and self.orig_loss:
            printed_res = f"loss: {loss}"
            if "min_expected_loss" in kwargs and kwargs['min_expected_loss'] is not None:
                printed_res = printed_res + f", min_exp_loss: {kwargs['min_expected_loss']}"
            printed_res = printed_res + f", rtf_loss: {rtf_loss}, audio_loss: {audio_loss}, auc_score: {auc_score}"
            print(printed_res)

        if not return_dict:
            output = (logits_per_audio, logits_per_rtf, rtf_embeds, audio_embeds, audio_outputs,
                      cosine_sim, auc_score)
            return ((loss,) + output) if loss is not None else output

        return CarirOutput(
            loss=loss,
            logits_per_audio=logits_per_audio,
            logits_per_rtf=logits_per_rtf,
            rtf_embeds=rtf_embeds,
            audio_embeds=audio_embeds,
            rtf_model_output=None,
            audio_model_output=audio_outputs,
            cosine_sim = cosine_sim,
            auc_score = auc_score
        )

    def get_auc(self, sim):
        with torch.no_grad():
            auc_score = CarirModel.contrastive_roc_auc_score(sim)
        return auc_score

    @staticmethod
    def contrastive_roc_auc_score(sim):
        mask = torch.eye(sim.shape[0], dtype=torch.bool)
        match = sim[mask]
        unmatch = sim[~mask]
        scores = torch.cat([match, unmatch])
        gt = torch.cat([torch.ones(match.size()), torch.zeros(unmatch.size())])
        return roc_auc_score(gt.bool().detach().cpu().numpy(), scores.detach().cpu().numpy())



def exmple_run():
    pretrained_dir = os.path.expanduser("~/.cache/huggingface/hub/models--carir/snapshots/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a")

    from .utils import example_run
    from .processing_carir import CarirProcessor

    fs = 8000
    reverbed_audio, h, md_dict = example_run()
    # carir_processor = CarirProcessor.from_json_path(os.path.join(pretrained_dir, "preprocessor_config.json"))
    carir_processor = CarirProcessor.from_pretrained(pretrained_dir)
    out_dict = carir_processor(h,
                               reverbed_audio,
                               return_tensors='pt')

    model = CarirModel.from_pretrained(pretrained_dir)

    outs = model(return_loss=True, **out_dict)
    print('finished running a single example')
