import os
import json
import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.audio_spectrogram_transformer.configuration_audio_spectrogram_transformer import \
    ASTConfig

from .instruct_carir import CarirV2, PretrainedAST
from .modeling_carir import RtfModel, CarirProjectionLayer, CarirModel, CarirRtfModel, CarirRtfConfig


class RtfClassifierConfig:
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_size,
                 projection_dim,
                 projection_hidden_act,
                 num_classes,
                 num_mid_layers,
                 freeze_encoder,
                 encoder_from_pretrain,
                 dummy_batches,
                 batch_size,
                 lr,
                 head_size=None,
                 **kwargs,
                 ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.num_classes = num_classes
        self.num_mid_layers = num_mid_layers
        self.freeze_encoder = freeze_encoder
        self.encoder_from_pretrain = encoder_from_pretrain
        self.dummy_batches = dummy_batches
        self.batch_size = batch_size
        self.lr = lr
        self.head_size = head_size

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


class MultiClassHead(nn.Module):
    def __init__(self, in_dim, out_dim, head_size=1):
        super(MultiClassHead, self).__init__()
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.BatchNorm1d(in_dim))
                                      for i in range(head_size)])
        self.linear2 = nn.Linear(in_dim, out_dim)

    def forward(self, emb):
        x = emb
        for i, l in enumerate(self.linears):
            x = l(x)
        out = self.linear2(x)
        return out


# class RtfClassifier(nn.Module):
#     def __init__(self, config):
#         super(RtfClassifier, self).__init__()
#         self.config = config
#
#         self.rtf2hidden_model = RtfModel(config)
#         if "/ast_data_v4/" not in self.config.encoder_from_pretrain:
#             self.rtf_projection = CarirProjectionLayer(config)
#         else:
#             self.rtf_projection = nn.Linear(self.config.hidden_size, self.config.projection_dim)  # backword_compatibility
#
#         self.num_classes = config.num_classes
#         if isinstance(self.num_classes, list):
#             self.classify = torch.nn.ModuleDict({str(i): MultiClassHead(config.projection_dim, self.num_classes[i]) for i in range(len(self.num_classes))})
#         else:
#             self.classify = torch.nn.Linear(config.projection_dim, config.num_classes)
#
#     def forward(self, input_rtf_features):
#         with torch.set_grad_enabled(not self.config.freeze_encoder):
#             hidden = self.rtf2hidden_model(rtf_inputs=input_rtf_features, return_dict=True)
#             proj_h = self.rtf_projection(hidden["rtf_features"])
#             proj_h = nn.functional.normalize(proj_h, dim=-1)
#
#         if isinstance(self.num_classes, list):
#             scores = [self.classify[str(i)](proj_h) for i in range(len(self.classify))]
#         else:
#             scores = self.classify(proj_h)
#
#         return scores
#
#     def fix_pretrain1_state_dict(self, state_dict):
#         dict_keys = list(state_dict.keys())
#         for k in dict_keys:
#             if k.startswith("audio") or k.startswith("logit_") or k.startswith("rtf_projection") \
#                     or k.startswith("embedder1") or k.startswith("embedder2_to_latents"):
#                 del state_dict[k]
#                 continue
#             new_name = k.replace('embedder2.model.', '')  # instruct carir
#             new_name = new_name.replace('rtf_model.model.', '')
#             if new_name != k:
#                 state_dict[new_name] = state_dict[k]
#                 del state_dict[k]
#             else:
#                 print(k)
#         return state_dict
#
#     def fix_pretrain2_state_dict(self, state_dict):
#         dict_keys = list(state_dict.keys())
#         for k in dict_keys:
#             if k.startswith("audio") or k.startswith("logit_") or k.startswith("rtf_model") or \
#                     k.startswith("embedder1") or k.startswith('embedder2.model.linears') or k.startswith('embedder2.model.last_linear'):
#                 del state_dict[k]
#                 continue
#             new_name = k.replace('embedder2_to_latents.', '')  # instruct carir
#             new_name = new_name.replace('rtf_projection.', '')
#             if new_name != k:
#                 state_dict[new_name] = state_dict[k]
#                 del state_dict[k]
#             else:
#                 print(k)
#         return state_dict
#
#     def load_encoder_from_pretrain(self, pretrain_path):
#         model_state_dict = torch.load(pretrain_path,
#                                       map_location=next(self.rtf2hidden_model.parameters()).device)
#         fixed_state_dict = self.fix_pretrain1_state_dict(model_state_dict)
#         self.rtf2hidden_model.load_state_dict(fixed_state_dict, strict=True)
#
#         model_state_dict = torch.load(pretrain_path,
#                                       map_location=next(self.rtf2hidden_model.parameters()).device)
#         fixed_state_dict = self.fix_pretrain2_state_dict(model_state_dict)
#         self.rtf_projection.load_state_dict(fixed_state_dict, strict=True)


class EmbClassifier(nn.Module):
    def __init__(self, model_path, projection_dim, num_classes, freeze_encoder = True, carir_processor=None, head_size=None):
        super(EmbClassifier, self).__init__()

        self.freeze_encoder = freeze_encoder
        if 'ast_' not in model_path and "dummy" not in model_path:
            self.carir_model = CarirModel.from_pretrained(model_path)
            self.max_audio_length = 500
        else:
            ast_config = ASTConfig(**json.load(open(os.path.join(model_path, "config.json"), "r")))
            embedder1 = PretrainedAST(carir_processor, ast_config)
            rtf_config = CarirRtfConfig.from_json_path(os.path.join(model_path, "config.json"))
            embedder2 = CarirRtfModel(rtf_config)
            self.carir_model = CarirV2(embedder1,
                                       embedder2,
                                       0.,
                                       0.,
                                       dim_latent=rtf_config.projection_dim,
                                       old_latent=False,  # TODO
                                       other_kwargs={'rtf_config': rtf_config},
                                       # dict_output_map = {'embedder1_latents': '', 'embedder2_latents': ''},
                                       )
            self.carir_model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu'))
            self.max_audio_length = ast_config.max_length

        self.num_classes = num_classes
        if isinstance(self.num_classes, list):
            self.classify = torch.nn.ModuleDict({str(i): MultiClassHead(projection_dim, self.num_classes[i]) for i in range(len(self.num_classes))})
        elif head_size is not None:
            self.classify = MultiClassHead(projection_dim, self.num_classes, head_size)
        else:
            self.classify = torch.nn.Linear(projection_dim, self.num_classes)

    def get_audio_emb(self, input_audio_features):
        with torch.set_grad_enabled(not self.freeze_encoder):
            # pad input_audio_features to max length:
            input_audio_features = torch.nn.functional.pad(input_audio_features, [0,0,0,self.max_audio_length-input_audio_features.shape[2]])
            proj_h = self.carir_model.get_embedder1_latents(input_audio_features, return_dict=True)
            return proj_h

    def scores_from_audio_features(self, input_audio_features):
        with torch.set_grad_enabled(not self.freeze_encoder):
            proj_h = self.get_audio_emb(input_audio_features)

        if isinstance(self.num_classes, list):
            scores = [self.classify[str(i)](proj_h) for i in range(len(self.classify))]
        else:
            scores = self.classify(proj_h)
        return scores

    def get_rtf_emb(self, input_rtf_features):
        with torch.set_grad_enabled(not self.freeze_encoder):
            proj_h = self.carir_model.get_embedder2_latents(input_rtf_features, return_dict=True)
            return proj_h

    def scores_from_rtf_features(self, input_rtf_features):
        with torch.set_grad_enabled(not self.freeze_encoder):
            proj_h = self.get_rtf_emb(input_rtf_features)

        if isinstance(self.num_classes, list):
            scores = [self.classify[str(i)](proj_h) for i in range(len(self.classify))]
        else:
            scores = self.classify(proj_h)
        return scores

    def forward(self, input_rtf_features=None, input_audio_features=None):
        if input_rtf_features is None:
            return self.scores_from_audio_features(input_audio_features)
        return self.scores_from_rtf_features(input_rtf_features)

    def load_classify_from_pretrain(self, pretrain_path):
        # remove unused parameters:
        state_dict = torch.load(pretrain_path, map_location=next(self.classify.parameters()).device)
        dict_keys = list(state_dict.keys())
        for k in dict_keys:
            if k.startswith("rtf2hidden_model") or k.startswith("rtf_projection") or k.startswith("carir_model"):
                del state_dict[k]
            else:
                new_name = k.replace('classify.', '')
                if new_name != k:
                    state_dict[new_name] = state_dict[k]
                    del state_dict[k]

        self.classify.load_state_dict(state_dict, strict=True)
