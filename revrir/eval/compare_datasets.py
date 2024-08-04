import pickle as pkl
import numpy as np
import torch

from .rooms_pred import roomsPred
from ..data.dataset_v2 import CARIR_dataset, collate_fn

from ..utils import get_generater_rirs_paths, get_audio_paths, get_audio


def get_model(model_path):
    model = roomsPred(model_path)
    return model

def get_dls(model, audios_dataset_name1, rir_dataset_name1, audios_dataset_name2, rir_dataset_name2):
    fs = model.get_fs()

    rir_paths1, md_paths1 = get_generater_rirs_paths(rir_dataset_name1)
    eval_audio_paths1 = get_audio_paths(audios_dataset_name1)[:200]

    eval_dataset1 = CARIR_dataset(audio_paths=eval_audio_paths1,
                                 rir_paths=rir_paths1,
                                 md_paths=md_paths1,
                                 fs=fs,  # Hz
                                 mode="eval",
                                 reverb = audios_dataset_name1 != "real_recordings",
                                 )

    rir_paths2, md_paths2 = get_generater_rirs_paths(rir_dataset_name2)
    eval_audio_paths2 = get_audio_paths(audios_dataset_name2)[:200]

    eval_dataset2 = CARIR_dataset(audio_paths=eval_audio_paths2,
                                 rir_paths=rir_paths2,
                                 md_paths=md_paths2,
                                 fs=fs,  # Hz
                                 mode="eval",
                                 reverb = audios_dataset_name2 != "real_recordings",
                                 )

    # Create a data loader to batch and shuffle the data
    eval_dataset1.worker_init_function(0)  # init it to get dataset.max_rir_length
    eval_dataset2.worker_init_function(0)  # init it to get dataset.max_rir_length

    nsample_rir = model.get_nsample_rir()
    print(f"nsample_rir: {nsample_rir}")
    collate_fn_p = collate_fn(processor=model.carir_processor,
                              fs=fs,
                              nsample_rir=nsample_rir,
                              save_rms=True)

    eval_dataloader1 = torch.utils.data.DataLoader(eval_dataset1,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   batch_size=min(200, len(eval_dataset1)),
                                                   collate_fn=collate_fn_p,
                                                   drop_last=True,
                                                   worker_init_fn=eval_dataset1.worker_init_function)

    eval_dataloader2 = torch.utils.data.DataLoader(eval_dataset2,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   batch_size=min(50, len(eval_dataset2)),
                                                   collate_fn=collate_fn_p,
                                                   drop_last=True,
                                                   worker_init_fn=eval_dataset2.worker_init_function)

    return eval_dataloader1, eval_dataloader2


def compare():
    from ..utils import HOME

    model_path = f"/{HOME}/asr/scratch/carir/classifier_training/rooms/train_300K_v3_3_ckpt_90K/model_49.pkl"

    eval_audios_dataset = "real_recordings"  # "real_recordings"   libri-dev-clean
    rir_dataset_name = "ace"  # v3_ood_benchmark_3K    v3_benchmark_3K  v3.1_benchmark_3K
    eval_on_audio = True

    model = get_model(model_path)

    eval_dataloader1, eval_dataloader2 = get_dls(model, "libri-dev-clean", rir_dataset_name, "libri-dev-clean", "v3.2_benchmark_3K")
    pred_labels = []
    gt_labels = []
    audios = None
    for batch in eval_dataloader1:
        if eval_on_audio:
            pred_l = model(batch["input_audio_features"], batch["input_rtf_features"], batch['room_id'])
        else:
            pred_l = model(None, batch["input_rtf_features"], batch['room_id'])
        if audios is None:
            audios = batch['audios']
        else:
            audios = np.concatenate((audios,batch['audios']))

        pred_labels = pred_labels + pred_l.tolist()

        import matplotlib.pyplot as plt
        b2 = next(iter(eval_dataloader2))

        speech_size_by_label = {0: [], 1: [], 2: []}
        speech_rms_by_label = {0: [], 1: [], 2: []}
        for i, room_id in enumerate(batch['room_id']):
            speech_size_by_label[room_id].append(batch['speech_size'][i])
            speech_rms_by_label[room_id].append(batch['speech_rms'][i])

        plt.scatter(speech_rms_by_label[0], speech_size_by_label[0], c='r', label=f'{rir_dataset_name}_small')
        plt.scatter(speech_rms_by_label[1], speech_size_by_label[1], c='g', label=f'{rir_dataset_name}_large')
        plt.scatter(speech_rms_by_label[2], speech_size_by_label[2], c='b', label=f'{rir_dataset_name}_hall')

        speech_size_by_label = {0: [], 1: [], 2: []}
        speech_rms_by_label = {0: [], 1: [], 2: []}
        for i, room_id in enumerate(b2['room_id']):
            speech_size_by_label[room_id].append(b2['speech_size'][i])
            speech_rms_by_label[room_id].append(b2['speech_rms'][i])

        plt.scatter(speech_rms_by_label[0], speech_size_by_label[0], c='r', label='audio1_small', marker='x')
        plt.scatter(speech_rms_by_label[1], speech_size_by_label[1], c='g', label='audio1_large', marker='x')
        plt.scatter(speech_rms_by_label[2], speech_size_by_label[2], c='b', label='audio1_hall', marker='x')

        plt.legend()
        plt.show()

        gt_l = batch['room_id']
        gt_labels = gt_labels + gt_l
        print(np.mean(gt_labels == np.array(pred_labels, dtype=np.int32)))
        if len(gt_labels) > 200:
            break
    pred_labels = np.array(pred_labels, dtype=np.int32)


    cm = confusion_matrix(gt_labels, pred_labels, normalize='true')
    acc = np.mean(gt_labels == pred_labels)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="hot")
    plt.xticks(ticks=range(3), labels=['small', 'large', 'hall'])
    plt.xlabel('Pred')
    plt.yticks(ticks=range(3), labels=['small', 'large', 'hall'])
    plt.ylabel('GT')
    plt.title(f'Audios dataset: {eval_audios_dataset}'
              f'\nRIR dataset: {rir_dataset_name}'
              f'\nResults on {"audio" if eval_on_audio else "rir"} embeddings'
              f'\n# samples: {len(gt_labels)}'
              f'\nAccuracy: {acc * 100:0.2f}%')
    plt.colorbar()
    plt.show()


def run():
    compare()


if __name__ == '__main__':
    run()