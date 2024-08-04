import os
import pickle as pkl
import numpy as np
import torch
from ..utils import HOME

BM_PATH = f'/{HOME}/asr/scratch/carir/benchmarks/room_classifier/v3_1'

name2id = {'small': 0, 'large': 1, 'hall': 2}


def generate_benchmark():
    from ..scripts.generate_rirs import generate_n_rirs_v3
    generate_n_rirs_v3(100, L=[3, 4, 3], desc='small', save_dir=BM_PATH, version="v3")
    generate_n_rirs_v3(100, L=[10, 9, 3], desc='large', save_dir=BM_PATH, version="v3")
    generate_n_rirs_v3(100, L=[2, 10, 2.5], desc='hall', save_dir=BM_PATH, version="v3")


def collect_benchmark_data():
    pkls_path = [os.path.join(BM_PATH, x) for x in sorted(os.listdir(BM_PATH)) if x.endswith('.pkl')]
    gt_labels = []
    rirs = []
    for pkl_p in pkls_path:
        room_type = os.path.basename(pkl_p).split('.')[0].split('_')[1]
        v, max_size, room_rirs = pkl.load(open(pkl_p, 'rb'))
        gt_labels.extend([{'room_id': name2id[room_type],
                           'rir_path': pkl_p,
                           'rir_md_path': pkl_p.replace('.pkl', '_md.csv'),
                           'rir_indx': i} for i in range(len(room_rirs))])
        rirs.extend(room_rirs)
    assert len(gt_labels) == len(rirs)
    from collections import Counter
    print('Dataset Stats')
    print(Counter([x['room_id'] for x in gt_labels]))

    save_path = os.path.join(BM_PATH, 'bm_data.pkl')
    pkl.dump((gt_labels, rirs), open(save_path, 'wb'), protocol=-1)
    print('finished generating bm data')


def get_model(model_path):
    from ..models.classifier import RtfClassifierConfig, RtfClassifier
    from ..feature_extraction_carir import RtfFeatureExtractor
    from ..data.classify_datasets import collate_fn
    import json

    config_path = os.path.join(os.path.dirname(model_path), "config.conf")
    config = RtfClassifierConfig.from_dict(json.load(open(config_path, "r")))

    model = RtfClassifier(config) # TODO: change to roomspred
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    feature_extractor = RtfFeatureExtractor(feature_size=-1, sampling_rate=8000, padding_value=0)
    nsample_rir = (config.in_dim - 1) * 2
    print(f"nsample_rir: {nsample_rir}")
    collate_fn_p = collate_fn(feature_extractor=feature_extractor,
                              nsample_rir=nsample_rir)

    def run_model(rir):
        with torch.no_grad():
            model_input = collate_fn_p([{"RIR": rir, "gt": 0}])['input_rtf_features']
            scores = model(model_input[None, :])
            return scores.argmax(-1).item()

    return run_model


def evalaute(eval_from_audio = False):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    gt_labels, rirs = pkl.load(open(os.path.join(BM_PATH, 'bm_data.pkl'), 'rb'))
    model_path = f"/{HOME}/asr/scratch/carir/classifier_training/rooms/train_300K_v3_ckpt_300K/model_49.pkl"
    if eval_from_audio:
        from .eval_classification_from_audio import get_model as get_emb_model
        model, _ = get_emb_model(model_path)
        pred_labels = [model(None, rir) for rir in rirs]
    else:
        model = get_model(model_path)
        pred_labels = [model(rir) for rir in rirs]
    gt_labels = np.array([x['room_id'] for x in gt_labels], dtype=np.int32)
    pred_labels = np.array(pred_labels, dtype=np.int32)

    cm = confusion_matrix(gt_labels, pred_labels, normalize='true')
    acc = np.mean(gt_labels == pred_labels)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.xticks(ticks=range(3), labels=['small', 'large', 'hall'])
    plt.xlabel('Pred')
    plt.yticks(ticks=range(3), labels=['small', 'large', 'hall'])
    plt.ylabel('GT')
    plt.title(f'Confusion Matrix: {acc * 100:0.2f}% Acc.')
    plt.colorbar()
    plt.show()


def run():
    # generate_benchmark()
    # collect_benchmark_data()
    evalaute()


if __name__ == '__main__':
    run()