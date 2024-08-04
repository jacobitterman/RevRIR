import torch
import random
import numpy as np
import pickle as pkl

from .rooms_pred import roomsPred
from ..data.dataset_v2 import CARIR_dataset, collate_fn

from ..utils import get_generater_rirs_paths, get_audio_paths, get_audio



def get_model(model_path, audios_dataset_name, rir_dataset_name, speakers=-1):
    model = roomsPred(model_path)

    rir_paths, md_paths = get_generater_rirs_paths(rir_dataset_name)
    eval_audio_paths = get_audio_paths(audios_dataset_name)
    if isinstance(speakers, int) and speakers > 0:
        eval_speakers = list(dict.fromkeys([s.split("/")[-3] for s in eval_audio_paths]))
        random.shuffle(eval_speakers)
        eval_speakers = eval_speakers[:speakers]
        print(f"chose speakers: {eval_speakers}")
        eval_audio_paths = [s for s in eval_audio_paths if s.split("/")[-3] in eval_speakers] * 10
    elif isinstance(speakers, str):
        print(f"chose speakers: {speakers}")
        eval_audio_paths = [s for s in eval_audio_paths if s.split("/")[-3] == speakers] * 10

    fs = model.get_fs()

    eval_dataset = CARIR_dataset(audio_paths=eval_audio_paths,
                                 rir_paths=rir_paths,
                                 md_paths=md_paths,
                                 fs=fs,  # Hz
                                 mode="eval",
                                 reverb = audios_dataset_name != "real_recordings",
                                 )

    # Create a data loader to batch and shuffle the data
    eval_dataset.worker_init_function(0)  # init it to get dataset.max_rir_length
    nsample_rir = model.get_nsample_rir()
    print(f"nsample_rir: {nsample_rir}")
    collate_fn_p = collate_fn(processor=model.carir_processor,
                              fs=fs,
                              nsample_rir=nsample_rir,
                              )

    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                  shuffle=True,
                                                  num_workers = 0,
                                                  batch_size = min(50, len(eval_dataset)),
                                                  collate_fn=collate_fn_p,
                                                  drop_last=True,
                                                  worker_init_fn=eval_dataset.worker_init_function)

    return model, eval_dataloader


def evalaute():
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    from ..utils import HOME
    # model_path = f"/{HOME}/asr/scratch/carir/classifier_training/rooms/train_300K_v3_2_ckpt_68K_from_openslr/model_49.pkl"  # best on 3 rooms - 87%
    model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_bins_ckpt_350/model_50.pkl"   # 83% on rir, 37% on audio
    # model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_ast_no_4_ckpt_350/model_50.pkl"  # 78% on rir, 34% on audio
    # model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_1_compression_ckpt_350/model_26.pkl"
    # model_path = "/orcam/asr/scratch/carir/classifier_training/110_rooms/v4_1_comp_dual_loss2_ckpt_340_unfreeze_encoder/model_1.pkl"
    # model_path = "/orcam/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_ast_no_4_ckpt_350_on_audio/model_9.pkl"


    eval_audios_dataset = "libri-dev-clean"  # real_recordings   libri-dev-clean
    rir_dataset_name = "v4_benchmark_3K"  # v3_ood_benchmark_3K  v3.2_benchmark_3K  ace  v4_benchmark_3K
    eval_on_audio = True

    model, eval_dataloader = get_model(model_path, eval_audios_dataset, rir_dataset_name, speakers=-1)
    pred_labels = []
    gt_labels = []
    scores = []
    print(f"len(eval_dataloader): {len(eval_dataloader)}")
    for batch in eval_dataloader:
        if eval_on_audio:
            pred_l = model(batch["input_audio_features"], batch["input_rtf_features"])
        else:
            pred_l = model(None, batch["input_rtf_features"])
        pred_labels = pred_labels + pred_l.tolist()

        scores = scores + model.scores.tolist()

        gt_l = batch['v4_room_indx'] if rir_dataset_name.startswith("v4") else batch['room_id']
        gt_labels = gt_labels + gt_l
        print(np.mean(gt_labels == np.array(pred_labels, dtype=np.int32)))
        if len(gt_labels) >= 400:
            break
    pred_labels = np.array(pred_labels, dtype=np.int32)

    scores = np.array(scores)
    gt_idxs = np.array(gt_labels, dtype=np.int32)
    ranks = scores.shape[1] - ((scores.argsort(-1) == gt_idxs[:,None]).nonzero()[1] / 1.0)
    room_ranks = []
    for i in range(scores.shape[1]):
        room_rank = ranks[gt_idxs==i].mean()
        room_ranks.append(room_rank)
    room_ranks = np.array(room_ranks)

    cm = confusion_matrix(gt_labels, pred_labels, normalize='true')
    acc = np.mean(gt_labels == pred_labels)

    num_labels = len(model.rooms)
    ticks_labels = [f"{i}" if i%5 == 0 else '' for i in range(num_labels)] if num_labels > 10 else model.rooms.keys()

    # Plot confusion matrix using seaborn
    fig, axs = plt.subplots(1, 2, figsize=(5, 5), sharex=True, sharey=True)
    im1 = axs[1].imshow(np.repeat(room_ranks[:, None], cm.shape[0], axis=1), vmin=1, vmax= 10 if eval_on_audio else 5)
    axs[1].set_xlabel('Rank')
    axs[1].set_ylabel('Room')
    axs[1].set_xticks(ticks=range(num_labels), labels=ticks_labels)
    plt.colorbar(im1, ax=axs[1])

    im0 = axs[0].imshow(cm, cmap="hot", vmin=0, vmax=1)
    axs[0].set_xticks(ticks=range(num_labels), labels=ticks_labels)
    axs[0].set_xlabel('Pred')
    axs[0].set_yticks(ticks=range(num_labels), labels=ticks_labels)
    axs[0].set_ylabel('GT')
    if num_labels > 10:
        rooms = [16, 52, 42]
        axs[0].plot((0, num_labels), (rooms[0], rooms[0]), 'y')
        axs[0].plot((rooms[0], rooms[0]), (0, num_labels), 'y')
        axs[0].plot((0, num_labels), (rooms[0] + rooms[1], rooms[0] + rooms[1]), 'y')
        axs[0].plot((rooms[0] + rooms[1], rooms[0] + rooms[1]), (0, num_labels), 'y')
        axs[0].set_xlim([0, cm.shape[0]])
        axs[0].set_ylim([0, cm.shape[1]][::-1])

        axs[1].plot((0, num_labels), (rooms[0], rooms[0]), 'r')
        axs[1].plot((0, num_labels), (rooms[0] + rooms[1], rooms[0] + rooms[1]), 'r')
        axs[1].set_xlim([0, cm.shape[0]])
        axs[1].set_ylim([0, cm.shape[1]][::-1])

    title = f'Audios dataset: {eval_audios_dataset}'
    if eval_audios_dataset != "real_recordings":
        title = title + f'\nRIR dataset: {rir_dataset_name}'
    if eval_on_audio:
        title = title + f'\nResults on audio embeddings'
    else:
        title = title + f'\nResults on rir embeddings'
    title = title + f'\n# samples: {len(gt_labels)} \nAccuracy: {acc * 100:0.2f}%, Av.rank: {ranks.mean():0.2f}'
    fig.suptitle(title)

    plt.colorbar(im0, ax=axs[0])
    plt.show()


def run():
    # generate_benchmark()
    # collect_benchmark_data()
    evalaute()


if __name__ == '__main__':
    run()