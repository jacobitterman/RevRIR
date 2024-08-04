import os
import json
import torch
import numpy as np
import pandas as pd
import pickle as pkl

from ..models.classifier import RtfClassifierConfig, EmbClassifier
from ..processing_carir import CarirProcessor
from ..utils import pad_array

from .classifier_benchmark import name2id



class roomsPred():
    def __init__(self, model_path, forced_encoder_path = None, use_cuda = False,):
        config_path = os.path.join(os.path.dirname(model_path), "config.conf")
        config = RtfClassifierConfig.from_dict(json.load(open(config_path, "r")))
        if forced_encoder_path is not None:
            config.encoder_from_pretrain = forced_encoder_path

        self.carir_processor = CarirProcessor.from_pretrained(config.encoder_from_pretrain)

        model = EmbClassifier(config.encoder_from_pretrain, config.projection_dim, config.num_classes,
                              carir_processor=self.carir_processor, head_size=config.head_size)
        model.load_classify_from_pretrain(model_path)
        model.eval()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            model = model.cuda()

        self.config = config
        self.model = model

        # id to room:
        rooms_path = os.path.join(os.path.dirname(model_path), "rooms.pkl")
        self.rooms =  pkl.load(open(rooms_path, 'rb')) if os.path.exists(rooms_path) else {0: 'small', 1: 'large', 2: 'hall'}
        self.scores = None

    def get_fs(self):
        return self.carir_processor.feature_extractor.sampling_rate

    def get_nsample_rir(self):
        return (self.model.carir_model.rtf_config.in_dim - 1) * 2

    def __call__(self, input_audio_features = None, input_rtf_features = None, *args, **kwargs):
        if self.use_cuda:
            input_audio_features = input_audio_features.cuda() if input_audio_features is not None else None
            input_rtf_features = input_rtf_features.cuda() if input_rtf_features is not None else None
        with torch.no_grad():
            if input_audio_features is not None:
                self.scores = self.model.scores_from_audio_features(input_audio_features)
            else:
                self.scores = self.model.scores_from_rtf_features(input_rtf_features)
            return self.scores.argmax(-1)

    def get_rtf_emb(self, input_rtf_features):
        if self.use_cuda:
            input_rtf_features = input_rtf_features.cuda()
        return self.model.get_rtf_emb(input_rtf_features)

    def get_audio_emb(self, input_audio_features):
        if self.use_cuda:
            input_audio_features = input_audio_features.cuda()
        return self.model.get_audio_emb(input_audio_features)

    def get_rtf_emb_from_rir(self, rir):
        rtf = pad_array(rir, self.get_nsample_rir())
        input_rtf_features = self.carir_processor(rtfs = [rtf], return_tensors='pt')['input_rtf_features']
        return self.get_rtf_emb(input_rtf_features)

    def get_audio_emb_from_audio(self, audio):
        audios = audio if audio.ndim == 2 else [audio]
        input_audio_features = self.carir_processor(audios=audios, return_tensors='pt')['input_features']
        return self.get_audio_emb(input_audio_features)

    def scores_from_audio(self, audio):
        audio_emb = self.get_audio_emb_from_audio(audio)
        self.scores = self.model.classify(audio_emb)
        return self.scores

    def classify_from_audio(self, audio):
        self.scores = self.scores_from_audio(audio)
        return self.scores.argmax(-1)

    def classify_from_audio_in_chunks(self, audio, chunk_size_sec = 10):
        audios = self.audio_to_chunks(audio, chunk_size=int(self.get_fs()*chunk_size_sec))
        self.scores = self.scores_from_audio(audios)
        return self.scores.argmax(-1)

    @staticmethod
    def classes_majority_vote(scores):
        values, counts = np.unique(scores, return_counts=True)
        ind = np.argmax(counts)
        return values[ind], counts

    @staticmethod
    def audio_to_chunks(audio, chunk_size):
        audio_len = len(audio)
        audios = []
        for i in range(0, audio_len, chunk_size):
            is_last = i + chunk_size >= audio_len
            if is_last:
                audios.append(np.concatenate((audio[i:i + chunk_size],
                                              np.zeros((i + chunk_size) - audio_len, dtype=audio.dtype))))
            else:
                audios.append(audio[i:i + chunk_size])
        return np.array(audios)

    def label_to_type(self, label):
        if label in self.rooms:
            return self.rooms[label]
        return None


def create_embeddings():
    from ..utils import HOME
    model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_bins_ckpt_350/model_49.pkl"
    # model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_ast_ckpt_200/model_2.pkl"
    model = roomsPred(model_path, use_cuda=True)

    audios_dataset_name = "libri-dev-clean"  # "real_recordings"   libri-dev-clean
    rir_dataset_name = "v4_benchmark_3K"  # v3_ood_benchmark_3K    v3_benchmark_3K  v3.1_benchmark_3K

    from ..utils import get_generater_rirs_paths, get_audio_paths, get_audio
    rir_paths, md_paths = get_generater_rirs_paths(rir_dataset_name)
    eval_audio_paths = get_audio_paths(audios_dataset_name)
    import random
    random.shuffle(eval_audio_paths)

    # speakers = ["7976", "3081"]
    # eval_audio_paths = [s for s in eval_audio_paths if s.split("/")[-3] in speakers]

    import scipy.signal as ss
    import pickle as pkl
    fs = model.get_fs()
    res = []
    rooms_idx = 0
    for room_path, room_md_path in zip(rir_paths, md_paths):
        room_type = os.path.basename(room_path).split('.')[0]
        print(f"starting room {room_type}")
        md_data = pd.read_csv(room_md_path)
        pkl_data = pkl.load(open(room_path, 'rb'))
        if len(pkl_data) == 3:
            version, max_size, rirs_all = pkl_data
            rooms = [1]
        else:
            version, max_size, rooms, n_samples_per_room, rirs_all = pkl_data
        for j, room in enumerate(rooms):
            print(f"starting room # {j + rooms_idx}")
            rirs = rirs_all[j*n_samples_per_room:]
            assert n_samples_per_room >= 5
            for i, rir in enumerate(rirs[:5]):
                room_md = md_data.iloc[j*n_samples_per_room + i].to_dict()
                rir_emb = model.get_rtf_emb_from_rir(rir)[0].cpu()
                print(f"starting rir {i}")
                for audio_path in eval_audio_paths[:50]:
                    # print(audio_path)
                    audio = get_audio(audio_path, fs)[:10 * fs]

                    reverb_audio = ss.convolve(rir, audio)[:10 * fs]  # TODO: assert max len
                    audio_emb = model.get_audio_emb_from_audio(reverb_audio).cpu()

                    speaker_id = audio_path.rsplit('/',4)[-3]
                    res.append({'speaker_id': speaker_id,
                                'room_type': room_type,
                                'audio_emb': audio_emb,
                                'rir_emb': rir_emb,
                                'audio_path': audio_path,
                                'room_md': room_md,
                                'rir_id': (j + rooms_idx) * n_samples_per_room + i,
                                'room_num': j + rooms_idx,
                                })

        rooms_idx += len(rooms)

    out_path = f'/homes/jacobb/tmp/v4_ckpt_350K_embeds_fixed_audio_len.pkl'
    out_path = "/tmp/v4_ckpt_350K_embeds_fixed_audio_len.pkl"
    pkl.dump(res, open(out_path, 'wb'))
    print(out_path)


if __name__ == '__main__':
    create_embeddings()
