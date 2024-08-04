import torch
import math
from dataclasses import dataclass
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torchaudio.transforms import MelSpectrogram, TimeStretch, FrequencyMasking, TimeMasking
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from transformers import AutoModel
from transformers import AutoProcessor, ASTModel
from transformers import AutoTokenizer
from x_clip.tokenizer import tokenizer
from functools import wraps
from typing import Optional, List


from .modeling_carir import CarirProjectionLayer


# https://github.com/lucidrains/musiclm-pytorch/blob/main/musiclm_pytorch/musiclm_pytorch.py


def exists(val):
    return val is not None

def round_down_nearest_multiple(n, divisor):
    return n // divisor * divisor

# decorators

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# tensor functions

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    pe = pe.type(dtype)

    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)


def contrastive_loss(logits: torch.Tensor, labels = None) -> torch.Tensor:
    if labels is None:
        labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)

# biasless layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x


def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim_hidden, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # qk rmsnorm, technique circulating within brain used to stabilize a 22B parameter vision model training

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout),
            ]))

    def forward(
        self,
        x,
        rel_pos_bias = None,
        mask = None
    ):

        for attn, ff in self.layers:
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            x = ff(x) + x

        return x


# Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778
def pair(t):
    return (t, t) if not isinstance(t, tuple) else t


class AudioSpectrogramTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        spec_sample_rate=44100,
        spec_n_fft = 1024,
        spec_win_length = 1024,
        spec_hop_length = 512,
        spec_n_mels=128,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        spec_f_min=50,
        spec_f_max=14000,
        patch_dropout_prob = 0.25,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.patch_size = pair(patch_size)
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]),
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, dim),
            nn.LayerNorm(dim)
        )

        # self.spec = Spectrogram(
        #     n_fft = spec_n_fft,
        #     power = spec_power,
        #     win_length = spec_win_length,
        #     hop_length = spec_hop_length,
        #     pad = spec_pad,
        #     center = spec_center,
        #     pad_mode = spec_pad_mode
        # )

        self.spec = MelSpectrogram(sample_rate=spec_sample_rate,
                                   n_fft=spec_n_fft,
                                   win_length=spec_win_length,
                                   hop_length=spec_hop_length,
                                   pad=spec_pad,
                                   center=spec_center,
                                   pad_mode=spec_pad_mode,
                                   n_mels=spec_n_mels,
                                   f_min=spec_f_min,
                                   f_max=spec_f_max,
                                   normalized=True)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=spec_n_fft, hop_length=spec_hop_length,
            win_length=spec_win_length, window='hann', center=spec_center, pad_mode=spec_pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=spec_sample_rate, n_fft=spec_n_fft,
            n_mels=spec_n_mels, fmin=spec_f_min, fmax=spec_f_max, ref=1.0, amin=1e-10, top_db=None,
            freeze_parameters=True)

        # SpecAugment - seems to be widely used in audio field https://arxiv.org/abs/1904.08779
        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True),
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask),
            TimeMasking(time_mask_param = spec_aug_time_mask),
        )

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_mult = ff_mult,
            ff_dropout = ff_dropout
        )

        self.norm = LayerNorm(dim)

        # patch dropout

        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias

        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        x,
        force_no_patch_dropout = False
    ):
        batch, device = x.shape[0], x.device

        # x = self.spec(x)
        x = self.logmel_extractor(self.spectrogram_extractor(x))
        x = rearrange(x, 'b a w h -> (b a) h w')
        x = x.flip(1)
        # if self.training:
        #     x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes

        height, width = x.shape[-2:]
        patch_height, patch_width = self.patch_size

        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        x = x[..., :rounded_height, :rounded_width]

        # to patches

        x = self.to_patch_tokens(x)

        # get number of patches along height and width

        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions

        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding

        x = x + posemb_sincos_2d(x)

        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout

        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            batch_indices = torch.arange(batch, device = device)
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            x = x[batch_indices, patch_indices_keep]

            grid = repeat(grid, '... -> b ...', b = batch)
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias

        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # attention, what else

        x = self.transformer(x, rel_pos_bias = rel_pos_bias)

        # final global average and norm (most recent papers show this is superior to CLS token)

        x = reduce(x, 'b n d -> b d', 'mean')

        return self.norm(x)


# text transformer
class PretrainedAST(nn.Module):
    def __init__(self, processor, pretrain_dir, augment=False, ignore_mismatched_sizes=False):
        super().__init__()
        self.pretrain_dir_or_config = pretrain_dir

        self.processor = processor
        if isinstance(pretrain_dir, str): # if it's path to model
            self.model = ASTModel.from_pretrained(pretrain_dir, ignore_mismatched_sizes=ignore_mismatched_sizes)
        else:  # if it's config
            self.model = ASTModel(pretrain_dir)
        self.model.train()
        self.dim=768
        self.augment = augment

    def forward(self, inputs):
        if inputs.ndim == 4 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        outputs = self.model(inputs)
        return outputs[1]

    def from_raw(self, x):
        inputs = self.processor(x, sampling_rate=16000, return_tensors="pt")
        if self.augment:
            with torch.no_grad():
                inputs['input_values'] = FrequencyMasking(freq_mask_param=100)\
                                        (TimeMasking(time_mask_param = 20)(inputs['input_values']))
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        return self.forward(inputs)



class TextTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
    ):
        super().__init__()
        self.dim = dim

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        self.cls_token = nn.Parameter(torch.randn(dim))

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        self.pad_id = pad_id
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None
    ):
        assert exists(x) ^ exists(raw_texts)

        if exists(raw_texts):
            x = tokenizer.tokenize(raw_texts).to(self.device)

        if not exists(mask):
            mask = x != self.pad_id

        b, n, device = *x.shape, x.device

        # token embedding + positional embedding

        x = self.token_emb(x)

        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        x = x + self.pos_emb(torch.arange(n, device = device))

        # cls tokens, as in bert

        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        x, ps = pack([cls_tokens, x], 'b * d')

        # account for attending to cls token with self attention mask

        mask = F.pad(mask, (1, 0), value = True)

        # attention

        x = self.transformer(x, mask = mask)

        # unpack the cls tokens

        cls_tokens, _ = unpack(x, ps, 'b * d')

        return self.norm(cls_tokens)


# main classes
class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear2 = nn.Linear(d_out, d_out, bias=False)
        self.layer_norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed1 = self.linear1(x)
        embed2 = self.drop(self.linear2(F.gelu(embed1)))
        embeds = self.layer_norm(embed1 + embed2)
        return embeds


class TextEncoder(nn.Module):
    def __init__(self, text_model: str, dim: int, text_max_length=64, outout_type='cls', freezed=False) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(text_model)
        if 't5' in text_model:
            self.base = self.base.encoder
        if freezed:
            for p in self.base.parameters():
                p.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.dim = dim
        self.text_max_length = text_max_length
        self.outout_type = outout_type
        # self.projection = Projection(transformer_embed_dim, d_out)

    def forward(self, x, **kwargs):
        if type(x) == str:
            x = self.tokenizer.encode_plus(text=x,
                                      max_length=self.text_max_length,
                                      truncation=True,
                                      padding='max_length',
                                      add_special_tokens=True,
                                      return_tensors="pt")
        out = self.base(**x).last_hidden_state
        if self.outout_type == 'cls':
            out = out[:, 0, :]  # get CLS token output
        elif self.outout_type == 'pooled':
            out = out.mean(1)
        # projected_vec = self.projection(out)
        # return projected_vec
        return out


@dataclass
class ProjectionConfig():
    projection_hidden_act = "relu"
    hidden_size = None
    projection_dim = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class CarirV2(nn.Module):
    def __init__(
            self,
            embedder1,
            embedder2,
            logit_scale_init_value,
            logit_scale_init2_value=0.5,
            dim_latent = 128,
            decoupled_contrastive_learning = True,
            calc_auc = False,
            projection_hidden_act="relu",
            old_latent = False,
            other_kwargs = None,
            dict_output_map = None,
    ):
        super().__init__()
        self.dim_latent = dim_latent

        self.embedder1 = embedder1
        self.embedder2 = embedder2

        if old_latent:
            self.embedder1_to_latents = nn.Linear(self.embedder1.dim, dim_latent)
            self.embedder2_to_latents = nn.Linear(self.embedder2.dim, dim_latent)
        else:
            p = ProjectionConfig(projection_hidden_act=projection_hidden_act,
                                 hidden_size=self.embedder1.dim, projection_dim=dim_latent)
            self.embedder1_to_latents = CarirProjectionLayer(p)
            p = ProjectionConfig(projection_hidden_act=projection_hidden_act,
                                 hidden_size=self.embedder2.dim, projection_dim=dim_latent)
            self.embedder2_to_latents = CarirProjectionLayer(p)

        self.logit_scale_a = nn.Parameter(torch.tensor(logit_scale_init_value))
        self.logit_scale_r = nn.Parameter(torch.tensor(logit_scale_init2_value))

        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.calc_auc = calc_auc
        self.scales_diff = logit_scale_init_value != logit_scale_init2_value

        if other_kwargs is not None and isinstance(other_kwargs, dict):
            self.__dict__.update(other_kwargs)

        self.dict_output_map = dict_output_map

    def get_embedder_latents(self, embedder, latenter, embedder_inputs):
        embeds = embedder(embedder_inputs)
        latents = latenter(embeds)
        return l2norm(latents)

    def get_embedder1_latents(self, embedder1_inputs, return_dict=False):
        res = self.get_embedder_latents(self.embedder1, self.embedder1_to_latents, embedder1_inputs)
        if return_dict and self.dict_output_map:
            return {self.dict_output_map['embedder1_latents']: res}
        return res

    def get_embedder2_latents(self, embedder2_inputs, return_dict=False):
        res = self.get_embedder_latents(self.embedder2, self.embedder2_to_latents, embedder2_inputs)
        if return_dict and self.dict_output_map:
            return {self.dict_output_map['embedder2_latents']: res}
        return res

    # def gather_latents_from_all_devices(self, embedder1_latents, embedder2_latents, labels):
    #     world_size, device = torch.distributed.get_world_size(), torch.device(f'cuda:{torch.distributed.get_rank()}')
    #     with torch.no_grad():
    #         embedder1_latents_gathered = [torch.zeros(embedder1_latents.shape[0],
    #                                               self.dim_latent, device=device) for _ in range(world_size)]
    #         torch.distributed.all_gather(embedder1_latents_gathered, embedder1_latents)
    #
    #         embedder2_latents_gathered = [torch.zeros(embedder2_latents.shape[0],
    #                                              self.dim_latent, device=device) for _ in range(world_size)]
    #         torch.distributed.all_gather(text_gathered_latents, embedder2_latents)
    #
    #         labels_gathered = [torch.zeros(labels.shape[0], 1, dtype=torch.int64, device=device) for _ in range(world_size)]
    #         torch.distributed.all_gather(labels_gathered, labels)
    #
    #     embedder1_latents_gathered[torch.distributed.get_rank()] = embedder1_latents
    #     embedder2_latents_gathered[torch.distributed.get_rank()] = embedder2_latents
    #     embedder1_latents = torch.vstack(embedder1_latents_gathered)
    #     embedder2_latents = torch.vstack(embedder2_latents_gathered)
    #     labels_gathered = torch.vstack(labels_gathered)
    #
    #     return embedder1_latents, embedder2_latents, labels_gathered

    def forward(
            self,
            input_em_1,
            input_em_2,
            labels,
            return_similarities = False,
            is_distributed=False,
            return_loss=True,
            **kwargs
    ):
        embedder1_latents_normed = self.get_embedder1_latents(input_em_1)
        embedder2_latents_normed = self.get_embedder2_latents(input_em_2)

        if is_distributed:
            embedder1_latents_normed, embedder2_latents_normed, labels = \
                self.gather_latents_from_all_devices(embedder1_latents_normed, embedder2_latents_normed, labels)

        cosine_sim = einsum('i d, j d -> i j', embedder1_latents_normed, embedder2_latents_normed)

        assert cosine_sim.shape[0] == cosine_sim.shape[1]

        if return_similarities:
            return cosine_sim

        loss = None
        # cosine similarity as logits
        logits_per_emb1 = torch.matmul(embedder1_latents_normed, embedder2_latents_normed.t()) * self.logit_scale_r.exp()
        logits_per_emb2 = torch.matmul(embedder2_latents_normed, embedder1_latents_normed.t()) * self.logit_scale_a.exp()
        if return_loss:
            loss1 = contrastive_loss(logits_per_emb1, labels)
            logits_for_second_loss = logits_per_emb2
            if self.scales_diff:
                logits_for_second_loss = logits_for_second_loss.t()
            loss2 = contrastive_loss(logits_for_second_loss, labels)
            loss = (loss1 + loss2) / 2.0

        auc_score = -1
        cosine_sim = None
        if self.calc_auc:
            cosine_sim = torch.einsum('i d, j d -> i j', embedder1_latents_normed, embedder2_latents_normed)
            auc_score = self.get_auc(cosine_sim)

        printed_res = f"total loss: {loss}"
        if "min_expected_loss" in kwargs and kwargs['min_expected_loss'] is not None:
            printed_res = printed_res + f", min_exp_loss: {kwargs['min_expected_loss']}"
        printed_res = printed_res + f", loss1: {loss1}, loss2: {loss2}, auc_score: {auc_score}"
        print(printed_res)

        return {'embedder1_latents_normed' : embedder1_latents_normed,
                'embedder2_latents_normed': embedder2_latents_normed,
                'loss': loss,
                'cosine_sim': cosine_sim,
                }
