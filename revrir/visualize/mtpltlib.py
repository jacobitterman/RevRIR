import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
import matplotlib.colors as mcolors
import matplotlib.markers as markers
from itertools import product

def plot_data(df, query, axs, plot_net=False, show_audio=True, show_rir=True):
    print(f'in plot_data: {query}')
    ax1, ax2, ax3, ax4 = axs
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax1.grid()
    default_markers = list(markers.MarkerStyle.markers.keys())[:7]
    # default_colors = list(mcolors.TABLEAU_COLORS.values())[:10]
    red_shades_rgb = [(x, 0, 0) for x in np.arange(1.0, 0.0, -0.25)]
    combinations_query = list(product(default_markers, red_shades_rgb))

    blue_shades_rgb = [(0, 0, x) for x in np.arange(0.8, 0.2, -0.15)]
    combinations_audio = list(product(default_markers, blue_shades_rgb))

    if show_rir:
        for j, (group_l, groupd_df) in enumerate(df.groupby(query)):
            marker, color = combinations_query[j % len(combinations_query)]
            ax1.scatter(groupd_df['x'], groupd_df['y'], color=color, marker=marker, label=group_l)
            if plot_net:
                assert s.source_type == 'speaker_sources_invariance'
                rir_id = list(set(groupd_df['rir_id_real']))[0]
                for x, y in zip(groupd_df['x'], groupd_df['y']):
                    ax1.plot([x, data_2d_audio[rir_id, 0]], [y, data_2d_audio[rir_id, 1]], color=color)
                # from IPython import embed; embed()
    if show_audio:
        for j, d in enumerate(data_2d_audio):
            marker, color = combinations_audio[j % len(combinations_audio)]
            ax1.scatter(d[0], d[1], color=color, label=f'audio_{j}', marker=marker, s=100)
    ax1.scatter(df.iloc[GT_indx]['x'], df.iloc[GT_indx]['y'], color='r', label='audio', marker='x', s=50)
    ax1.set_title(f'Interactive Scatter Plot Filter By: {query}')
    ax1.legend(prop={'size': 8}, ncols=3)

    ax2.plot(rirs[GT_indx, :])
    ax2.grid()
    ax2.set_ylabel('RIR- GT')
    ax3.grid()
    ax3.set_ylabel('RIR- Pred')
    ax4.plot(rtfs[GT_indx], label='GT RTF')
    ax4.grid()
    ax4.set_ylabel('RTFs')

def parse_data_from_tb_v3(tensors_path, md_path):
    import pandas as pd
    with open(tensors_path, 'rt') as fid:
        p_ls = fid.readlines()

    with open(md_path, 'rt') as fid:
        p_md = fid.readlines()
    assert len(p_md) == len(p_ls)
    batch_size = 55
    chunk_size = 2 * batch_size

    metadatas = [x.strip() for x in p_md]
    metadata_splits = [(metadatas[i:i + chunk_size // 2], metadatas[i + chunk_size // 2: i + chunk_size]) for i in range(0, len(metadatas), chunk_size)]
    metadata_audio, metadata_rtf = [sum(x, []) for x in list(zip(*metadata_splits))]
    min_val = np.minimum(len(metadata_audio), len(metadata_rtf))
    assert min_val % batch_size == 0
    metadata_audio = metadata_audio[:min_val]
    metadata_rtf = metadata_rtf[:min_val]
    assert all(y.startswith(x) for x, y in zip(metadata_audio, metadata_rtf)), 'audio and rtfs are not aligned'

    md_audio_list = []
    for speaker_str in metadata_audio:
        speaker_id, speaker_book, speaker_utter = speaker_str.split('_')
        md_audio_list.append({'speaker_id': speaker_id,
                              'speaker_book': speaker_book,
                              'speaker_utter': speaker_utter,})
    metadata_audio = pd.DataFrame(md_audio_list)

    md_rtf_list = []
    for metadata_rtf_str in metadata_rtf:
        speaker_str, room_str = metadata_rtf_str.split('_r_')
        speaker_id, speaker_book, speaker_utter = speaker_str.split('_')

        reciever_str, source_str = room_str.split('_s_')
        source_str, room_dim_str = source_str.split('_L_')
        room_dim_str = room_dim_str.split('_t60')[0]

        room_dim_list = eval(room_dim_str)
        try:
            source_list = eval(', '.join(source_str.split()))
        except:
            source_list = eval(', '.join(source_str.replace('[', '').replace(']', '').split()))
        try:
            reciever_list = eval(', '.join(reciever_str.replace('[[', '[').replace(']]', ']').split()))
        except:
            reciever_list = eval(', '.join(reciever_str.replace('[[ ', '[').replace(']]', ']').split()))
        md_rtf_list.append({'speaker_id': speaker_id,
                            'speaker_book': speaker_book,
                            'speaker_utter': speaker_utter,
                            'room_dim': room_dim_list,
                            'source': source_list,
                            'reciever': reciever_list,
                            })
    metadata_rtf = pd.DataFrame(md_rtf_list)

    embeddings = np.array([np.array([float(x.strip()) for x in p.split('\t')], dtype=np.float32) for p in p_ls])
    embeddings_splits = [(embeddings[i:i + chunk_size // 2],
                          embeddings[i + chunk_size // 2: i + chunk_size])
                         for i in range(0, len(embeddings), chunk_size)]
    embeddings_audio, embeddings_rtf = [np.vstack(x) for x in list(zip(*embeddings_splits))]
    embeddings_audio = embeddings_audio[:min_val]
    embeddings_rtf = embeddings_rtf[:min_val]
    return metadata_audio, metadata_rtf, embeddings_audio, embeddings_rtf


class State(object):
    query = ''
    plot_net = False
    source_type = ''
    show_audio = True
    show_rir = True

if __name__ == '__main__':

    import numpy as np
    import pandas as pd

    GT_indx = 10
    s = State()
    s.source_type = 'speaker_sources_invariance'
    if s.source_type == 'v3':
        p_t, p_md = "", ""
        metadata_audio, metadata_rtf, embeddings_audio, embeddings_rtf = parse_data_from_tb_v3(p_t, p_md)
        metadata_rtf['room_dim_str'] = metadata_rtf['room_dim'].apply(lambda x: str(x))

        audios = []
        rtfs = np.random.rand(len(metadata_audio), 1001)
        rirs = np.random.rand(len(metadata_audio), 2000)

        audio_emb = embeddings_audio[GT_indx, :].reshape(1, -1)
        print(metadata_audio.iloc[GT_indx])
        rtf_emb = embeddings_audio # embeddings_rtf
        df = metadata_rtf
        s.query = metadata_rtf.keys()[0]
        # from IPython import embed; embed()
    elif s.source_type == 'speaker_sources_invariance':
        import pickle as pkl
        a = pkl.load(open('/home/embed.pkl', 'rb'))

        audios = []
        rtfs = np.random.rand(500, 1001)
        rirs = np.random.rand(500, 2000)

        # Generate random high-dimensional data (128 dimensions)
        df = pd.DataFrame(a)

        # from IPython import embed; embed()
        rtf_emb = np.array([np.array(x).reshape(-1) for x in df['audio_emb'].values])
        print(f'there are {len(rtf_emb)} audio samples total')
        audio_emb = np.array([np.array(x).reshape(-1) for x in df['rir_emb'].values])
        df['rir_id_real'] = np.unique(audio_emb, axis=0, return_index=True, return_inverse=True)[-1]
        audio_emb = np.unique(audio_emb, axis=0)
        print(f'there are {len(audio_emb)} rirs')
        s.query = 'room_type'

    elif s.source_type == 'random_sources':
        audios = []
        rtfs = np.random.rand(500, 1001)
        rirs = np.random.rand(500, 2000)

        # Generate random high-dimensional data (128 dimensions)
        rtf_l = ['3_4_7'] * 30 + ['3_4_3'] * 30 + ['2_6_1'] * 30
        speaker_id_l = range(30)
        combinations_ = list(product(rtf_l, speaker_id_l))

        audio_emb = np.random.rand(1, 128)
        rtf_emb = np.random.rand(len(combinations_), 128)

        # from IPython import embed; embed()
        rtf_l, speaker_id_ = zip(*combinations_)
        df_dict = {'room_size': rtf_l,
                   'speaker_id': speaker_id_}
        df = pd.DataFrame(df_dict)
        s.query = 'room_size'
    elif s.source_type == 'random':
        audios = []
        rtfs = np.random.rand(500, 1001)
        rirs = np.random.rand(500, 2000)

        # Generate random high-dimensional data (128 dimensions)
        audio_emb = np.random.rand(1, 128)
        rtf_emb = np.random.rand(500, 128)
        rtf_l = ['3_4_7'] * 100 + ['3_4_3'] * 100 + ['2_6_1'] * 300
        df_dict = {'label': rtf_l,
                   'label2': rtf_l}
        df = pd.DataFrame(df_dict)
        s.query= 'label'
    else:
        assert 0
    # Apply t-SNE to reduce the dimensionality to 2D
    all_emb = np.vstack((audio_emb, rtf_emb))
    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(all_emb)

    data_2d_audio = data_2d[:len(audio_emb)].reshape(-1, 2)
    data_2d_rtf = data_2d[len(audio_emb):].reshape(-1, 2)

    assert len(df) == len(data_2d_rtf)
    df['x'] = data_2d_rtf[:, 0]
    df['y'] = data_2d_rtf[:, 1]

    # Create a scatter plot using Matplotlib
    fig = plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 3, 5)
    ax3 = plt.subplot(2, 3, 6, sharex=ax2, sharey=ax2)
    ax4 = plt.subplot(2, 3, 4)
    axs = [ax1, ax2, ax3, ax4]
    plot_data(df, query=s.query, axs=axs, plot_net=s.plot_net, show_audio=s.show_audio, show_rir=s.show_rir)


    global is_query
    is_query = True

    # Function to handle click events
    def on_click(event):
        # from IPython import embed; embed()
        global is_query
        if event.inaxes is not None:
            x_, y_ = event.xdata, event.ydata

            # find nearesr neighbour
            argmin_ind = np.argmin(np.abs(data_2d_rtf[:, 0] - x_) ** 2 + np.abs(data_2d_rtf[:, 1] - y_) ** 2)
            # ax.scatter(x_, y_, c='green', s=100, marker='x')
            ax3.cla()
            ax3.plot(rirs[argmin_ind, :])
            ax3.grid()
            ax4.cla()
            ax4.plot(rtfs[GT_indx, :], label='Pred RTF')
            ax4.plot(rtfs[argmin_ind, :], label='Pred RTF')
            ax4.grid()
            ax4.legend()
            if is_query:
                is_query = False
                ax1.scatter(data_2d_rtf[argmin_ind, 0], data_2d_rtf[argmin_ind, 1], c='m', s=100, marker='x', label='query')
                ax1.legend(prop={'size': 8}, ncols=3)
            else:
                ax1.scatter(data_2d_rtf[argmin_ind, 0], data_2d_rtf[argmin_ind, 1], c='m', s=100, marker='x')
            fig.show()


    def on_key(event):
        print(f'Key pressed: {event.key}')
        if event.key == 'h':
            show_popup_window()
        if event.key == 'n':
            s.plot_net = not s.plot_net
            plot_data(df, query=s.query, axs=axs, plot_net=s.plot_net, show_audio=s.show_audio, show_rir=s.show_rir)
            fig.show()
        if event.key == 'a':
            s.show_audio = not s.show_audio
            plot_data(df, query=s.query, axs=axs, plot_net=s.plot_net, show_audio=s.show_audio, show_rir=s.show_rir)
            fig.show()
        if event.key == 'r':
            s.show_rir = not s.show_rir
            plot_data(df, query=s.query, axs=axs, plot_net=s.plot_net, show_audio=s.show_audio, show_rir=s.show_rir)
            fig.show()


    def show_popup_window():
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        message_text = '\n'.join([f'{i}. {x}' for i, x in enumerate(sorted(list(df.keys())))])
        result = simpledialog.askstring('Possible Queries', f"{message_text}\nEnter text:")
        if result:
            messagebox.showinfo("Your Query: ", result)
        root.destroy()  # Destroy the hidden main window when done
        if result:
            if ',' in result:
                result = [x.strip() for x in result.split(',')]
            # from IPython import embed; embed()
            s.query = result
            plot_data(df, query=s.query, axs=axs, plot_net=s.plot_net, show_audio=s.show_audio, show_rir=s.show_rir)
            fig.show()
            print('finished plot')

    # Connect the click event to the scatter plot
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # show
    plt.show()