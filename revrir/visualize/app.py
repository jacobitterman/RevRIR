import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader


from ..data.dataset_v2 import collate_fn
from ..data.dataset_v3 import collate_fn_v3, CARIR_dataset_V3
from ..models.modeling_carir import CarirModel
from ..processing_carir import CarirProcessor
from ..utils import get_generater_rirs_paths, get_audio_paths, play_audio_ffplay
from ..data.dataset_v2 import CARIR_dataset as CARIR_dataset_v2

def app_from_random():
    audios = []
    rtfs = []
    # Generate random high-dimensional data (128 dimensions)
    audio_emb = np.random.rand(1, 128)
    rtf_emb = np.random.rand(500, 128)

    visualize_emb(audio_emb, rtf_emb, audios, rtfs)


def app_from_real_data(eval_dataset = 'libri-dev-clean',
                       model_path = '',
                       n_samples_eval = 200,
                       ):

    # load processor and model
    carir_processor = CarirProcessor.from_pretrained(model_path)
    carir_model = CarirModel.from_pretrained(model_path)

    rir_paths, md_paths = get_generater_rirs_paths("v3.2_benchmark_3K")
    eval_audio_paths = get_audio_paths(eval_dataset)[:n_samples_eval]
    print(f'selected {n_samples_eval} audio samples for eval, in practice we got {len(eval_audio_paths)}')
    fs = carir_processor.feature_extractor.sampling_rate

    eval_dataset = CARIR_dataset_V3(audio_paths=eval_audio_paths,
                                    rir_paths=rir_paths,
                                    md_paths=md_paths,
                                    fs=fs,                     # Hz
                                    mode = "eval",
                                    hard_negative_prob=.0,
                                    hard_positive_prob=.0,
                                    )

    # Create a data loader to batch and shuffle the data
    print(f'max_n_rir: {carir_model.rtf_config.in_dim}')
    nsample_rir = (carir_model.rtf_config.in_dim - 1) * 2
    eval_dataset.worker_init_function(0)  # init it to get dataset.max_rir_length
    assert nsample_rir >= eval_dataset.max_rir_length, eval_dataset.max_rir_length
    collate_fn_p = collate_fn_v3(processor=carir_processor,
                                 fs=fs,
                                 nsample_rir = nsample_rir,
                                 )


    dataloader = DataLoader(eval_dataset,
                            shuffle=True,
                            num_workers = 0,
                            batch_size = 50,
                            collate_fn=collate_fn_p,
                            drop_last=True,
                            worker_init_fn=eval_dataset.worker_init_function)

    dataloader_iter = iter(dataloader)
    model_input = next(dataloader_iter)
    with torch.no_grad():
        output = carir_model(**model_input)
        audio_emb, rtf_emb = output["audio_embeds"], output["rtf_embeds"]
        audios = model_input['audios']
        rtfs_md = model_input['rir_md']
        Ls = set([x['L'] for x in rtfs_md])
        rtfs_dict = {x: j for j, x in enumerate(Ls)}
        rtfs_l = [x['L'] for x in rtfs_md]
        from ..train.train_emb import classification_score
        rank, acc = classification_score(audio_emb, rtf_emb, np.arange(audio_emb.shape[0]))
        # for i in range(audio_emb.shape[0]):
        #     print(f"visualizing audio indx {i} from path {model_input['audio1_fpath'][i]}")
        #     visualize_emb(audio_emb[i: i + 5], rtf_emb[i: i + 5], rtf_emb, audios[i], rtfs_md, i, rtfs_l,
        #                   model_input["input_rtf_features"], model_input["RIR"])


def visualize_emb(audio_emb, audio_rtf_emb, rtf_emb, audio, rtf_md, GT_idx, rtfs_l, rtfs, rirs):
    all_emb = np.vstack((audio_emb, audio_rtf_emb, rtf_emb))
    # Apply t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(all_emb)
    data_2d_audio = data_2d[:len(audio_emb)].reshape(-1, 2)
    data_2d_audio_rtf = data_2d[len(audio_emb):len(audio_emb) + len(audio_rtf_emb)].reshape(-1, 2)
    data_2d_rtf = data_2d[len(audio_emb) + len(audio_rtf_emb):].reshape(-1, 2)

    df = pd.DataFrame({'x': data_2d_rtf[:, 0],
                       'y': data_2d_rtf[:, 1],
                       'label': rtfs_l})
    colors = 'rgb'

    # Create a scatter plot using Matplotlib
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 1, 1)
    ax1.grid()
    for j, (group_l, groupd_df) in enumerate(df.groupby('label')):
        ax1.scatter(groupd_df['x'], groupd_df['y'], color=colors[j], label=group_l)
    for i in range(len(data_2d_audio)):
        print(data_2d_audio[i, 0], data_2d_audio[i, 1])
        ax1.scatter(data_2d_audio[i, 0], data_2d_audio[i, 1], color='g', label='audio', marker='x')
    for i in range(len(data_2d_audio_rtf)):
        print(data_2d_audio_rtf[i, 0], data_2d_audio_rtf[i, 1])
        ax1.scatter(data_2d_audio_rtf[:, 0], data_2d_audio_rtf[:, 1], color='k', label='audio_rtf', marker='x')
    ax1.legend()
    ax1.set_title('Interactive Scatter Plot')
    ax2 = plt.subplot(2, 3, 5)
    ax2.plot(rirs[GT_idx])
    ax2.grid()
    ax2.set_ylabel('RIR- GT')
    ax3 = plt.subplot(2, 3, 6, sharex=ax2, sharey=ax2)
    ax3.set_ylabel('RIR- Pred')
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(rtfs[GT_idx], label='GT RTF')
    ax4.grid()
    ax4.set_ylabel('RTFs')
    global is_query
    is_query = True

    # Function to handle click events
    def on_click(event):
        global is_query
        if event.inaxes is not None:
            x_, y_ = event.xdata, event.ydata
            # find nearesr neighbour
            argmin_ind = np.argmin(np.abs(data_2d_rtf[:, 0] - x_) ** 2 + np.abs(data_2d_rtf[:, 1] - y_) ** 2)
            if is_query:
                is_query = False
                ax1.scatter(data_2d_rtf[argmin_ind, 0], data_2d_rtf[argmin_ind, 1], c='m', s=100, marker='x',
                            label='query')
                ax1.legend()
            else:
                ax1.scatter(data_2d_rtf[argmin_ind, 0], data_2d_rtf[argmin_ind, 1], c='m', s=100, marker='x')
            ax3.cla()
            ax3.plot(rirs[argmin_ind])
            ax3.grid()
            ax4.cla()
            ax4.plot(rtfs[GT_idx, :], label='Pred RTF')
            ax4.plot(rtfs[argmin_ind, :], label='Pred RTF')
            ax4.grid()
            ax4.legend()
            fig.show()
            print(f'GT rtf md: {rtf_md[GT_idx]}')
            print(f'GT audio_rtf md: {GT_idx}')
            print(f'rtf md: {rtf_md[argmin_ind]}')
            print(f'audio_rtf md: {argmin_ind}')
            # [x['L'] for x in rtfs_md]

    # Connect the click event to the scatter plot
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

import fire
if __name__ == '__main__':
    fire.Fire()