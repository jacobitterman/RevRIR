import torch
from torch import nn

from .modeling_carir import RtfModel, CarirProjectionLayer


class RtfModelConfig:
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_size,
                 projection_dim,
                 projection_hidden_act,
                 loss,
                 num_mid_layers,
                 freeze_encoder,
                 encoder_from_pretrain,
                 dummy_batches,
                 batch_size,
                 lr,
                 **kwargs,
                 ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.loss = loss
        self.num_mid_layers = num_mid_layers
        self.freeze_encoder = freeze_encoder
        self.encoder_from_pretrain = encoder_from_pretrain
        self.dummy_batches = dummy_batches
        self.batch_size = batch_size
        self.lr = lr

    def get_decoder_config(self):
        return RtfModelConfig(self.out_dim, self.in_dim, self.projection_dim, self.hidden_size,
                              self.projection_hidden_act, self.loss, self.num_mid_layers, self.freeze_encoder,
                              self.encoder_from_pretrain, self.dummy_batches, self.batch_size, self.lr)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


class EncoderDecoder(nn.Module):
    def __init__(self, config):
        super(EncoderDecoder, self).__init__()
        self.config = config

        self.rtf2hidden_model = RtfModel(config)
        self.rtf_project = CarirProjectionLayer(config)

        self.config_dec = self.config.get_decoder_config()

        self.rtf_deproj = CarirProjectionLayer(self.config_dec)
        self.rtf2fft_model = RtfModel(self.config_dec)

    def forward(self, input_rtf_features):
        with torch.set_grad_enabled(not self.config.freeze_encoder):
            hidden = self.rtf2hidden_model(rtf_inputs=input_rtf_features, return_dict=True)
            proj_h = self.rtf_project(hidden["rtf_features"])

        deproj_h = self.rtf_deproj(proj_h)
        rtf_fft = self.rtf2fft_model(deproj_h, return_dict=True)

        return rtf_fft

    def load_encoder_from_pretrain(self, pretrain_path):
        self.rtf2hidden_model.load_state_dict(torch.load(pretrain_path,
                                                         map_location=next(self.rtf2hidden_model.parameters()).device),
                                              strict=False)
        self.rtf_project.load_state_dict(torch.load(pretrain_path,
                                                    map_location=next(self.rtf_project.parameters()).device),
                                         strict=False)
