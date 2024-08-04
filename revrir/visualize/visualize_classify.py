import matplotlib.pyplot as plt
import numpy as np
import mplcursors
from tabulate import tabulate
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog

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

class State():
    prev_samples = None
    cur_samples = None
    n_neighbours = 10
    text = None
    scatter = []
    id2group_label = {}
    t2id = {}
    tsne_dist = False
    classify = True

def get_data_random():
    # Generate some sample data for two classes
    np.random.seed(42)
    num_points = 50

    # Class 1
    x_class1 = np.random.rand(num_points // 2) + 2
    y_class1 = np.random.rand(num_points // 2) + 2
    z_class1 = np.random.rand(num_points // 2) + 10

    # Class 2
    x_class2 = np.random.rand(num_points // 2)
    y_class2 = np.random.rand(num_points // 2)
    z_class2 = np.random.rand(num_points // 2)

    class_1_list = [{'x': x_class1,
                     'y': y_class1,
                     'volume': z_class1,
                     'room_type': ['3 4 3'] * len(x_class1),
                     'marker': 'o',
                     'label': 'RTF: small'},
                    {'x': x_class1 + 2,
                     'y': y_class1 + 2,
                     'volume': z_class1 - 3,
                     'room_type': ['9 10 3'] * len(x_class1),
                     'marker': '^',
                     'label': 'RTF: large'},
                    {'x': x_class1 - 2,
                     'y': y_class1 + 1,
                     'volume': z_class1 - 1,
                     'room_type': ['2 10 3'] * len(x_class1),
                     'marker': 'x',
                     'label': 'RTF: hall'}
                    ]
    class_2_list = [{'x': x_class2,
                     'y': y_class2,
                     'volume': z_class2,
                     'marker': '<',
                     'label': 'Audio'}]

    return class_1_list, class_2_list

def calc_room_type(room_dims):
    small = [(1.5, 4), (2.5, 4.5), (2, 3.5)]
    large = [(6, 13), (6,12), (2,5)]
    hall = [(1,3), (7,13), (2,3.5)]
    x, y, z = room_dims
    if small[0][0] <= x <= small[0][1] and small[1][0] <= y <= small[1][1] and small[2][0] <= z <= small[2][1]:
        return 'small'
    elif large[0][0] <= x <= large[0][1] and large[1][0] <= y <= large[1][1] and large[2][0] <= z <= large[2][1]:
        return 'large'
    elif hall[0][0] <= x <= hall[0][1] and hall[1][0] <= y <= hall[1][1] and hall[2][0] <= z <= hall[2][1]:
        return 'hall'
    else:
        assert 0

def get_real_data():
    from sklearn.manifold import TSNE

    #load tsv files
    p_t = '/home/Downloads/tensors_3.tsv'
    p_md = '/home/Downloads/metadata_3.tsv'
    metadata_audio, metadata_rtf, embeddings_audio, embeddings_rtf = parse_data_from_tb_v3(p_t, p_md)
    metadata_rtf['room_dim_str'] = metadata_rtf['room_dim'].apply(lambda x: str(x))
    metadata_rtf['volume'] = metadata_rtf['room_dim'].apply(lambda x: np.prod(x))
    metadata_rtf['room_type'] = metadata_rtf['room_dim'].apply(lambda x: calc_room_type(x))

    roomtype_2_marker = {'small': 'x',
                         'large': 'o',
                         'hall': '.'}
    metadata_rtf['marker'] = metadata_rtf['room_type'].apply(lambda x: roomtype_2_marker[x])

    # load Yaki's files
    load_yaki = True
    if load_yaki:
        import pickle as pkl
        import os
        metadata_audio = None
        real_audio_path = '/home/Downloads/real_audios_emb.pkl'
        pkl_data = pkl.load(open(real_audio_path, 'rb'))
        embeddings_audio = np.vstack(pkl_data[0])
        audio_emb = embeddings_audio
    else:
        audio_emb = embeddings_audio[:14, :]

    # Apply t-SNE to reduce the dimensionality to 2D
    rtf_emb = embeddings_rtf
    all_emb = np.vstack((audio_emb, rtf_emb))
    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(all_emb)

    data_2d_audio = data_2d[:len(audio_emb)].reshape(-1, 2)
    data_2d_rtf = data_2d[len(audio_emb):].reshape(-1, 2)
    # from IPython import embed; embed()

    inds_small = metadata_rtf[metadata_rtf['room_type'].apply(lambda x: x == 'small')].index.values
    inds_large = metadata_rtf[metadata_rtf['room_type'].apply(lambda x: x == 'large')].index.values
    inds_hall = metadata_rtf[metadata_rtf['room_type'].apply(lambda x: x == 'hall')].index.values
    assert len(inds_hall) + len(inds_large) + len(inds_small) == len(data_2d_rtf)

    class_1_list = [{'x': data_2d_rtf[inds_small, 0],
                     'y': data_2d_rtf[inds_small, 1],
                     'volume': metadata_rtf.iloc[inds_small]['volume'].values,
                     'room_type': metadata_rtf.iloc[inds_small]['room_dim'].values,
                     'marker': 'o',
                     'label': 'RTF: small',
                     'embed': rtf_emb[inds_small]},
                    {'x': data_2d_rtf[inds_large, 0],
                     'y': data_2d_rtf[inds_large, 1],
                     'volume': metadata_rtf.iloc[inds_large]['volume'].values,
                     'room_type': metadata_rtf.iloc[inds_large]['room_dim'].values,
                     'marker': '^',
                     'label': 'RTF: large',
                     'embed': rtf_emb[inds_large]},
                    {'x': data_2d_rtf[inds_hall, 0],
                     'y': data_2d_rtf[inds_hall, 1],
                     'volume': metadata_rtf.iloc[inds_hall]['volume'].values,
                     'room_type': metadata_rtf.iloc[inds_hall]['room_dim'].values,
                     'marker': 'x',
                     'label': 'RTF: hall',
                     'embed': rtf_emb[inds_hall]}
                    ]

    if load_yaki:
        fnames = [os.path.basename(x).rsplit('_chunk', 1)[0] for x in pkl_data[1]]
        class_2_list = []
        markers = '*s2+'
        room_types = sorted(set(fnames))
        t2v = {'small_shelter': 10,
               'hall': 42,
               'large': 96,
               'small': 27}
        assert len(markers) == len(room_types)
        assert len(data_2d_audio) == len(pkl_data[1])
        for room_idx, (room_type, marker) in enumerate(zip(room_types, markers)):
            s.t2id[room_type] = room_idx
            room_type_ind = [i for i, x in enumerate(fnames) if x == room_type]
            embeds = audio_emb[room_type_ind]
            # from IPython import embed; embed()
            class_2_list.append({'x': data_2d_audio[room_type_ind, 0],
                                 'y': data_2d_audio[room_type_ind, 1],
                                 'volume': np.array([t2v[room_type]] * len(room_type_ind)),
                                 'marker': marker,
                                 'label': f'Audio: {room_type}',
                                 'embed': embeds})

    else:
        class_2_list = [{'x': data_2d_audio[:, 0],
                         'y': data_2d_audio[:, 1],
                         'volume': np.arange(len(data_2d_audio)),
                         'marker': '<',
                         'label': 'Audio',
                         'embed': audio_emb}]

    return class_1_list, class_2_list

s = State()

# get data
class1_list, class2_list = get_real_data() # get_data_random() #

counter = 0
for i, class_ in enumerate(class1_list):
    for j in range(len(class_['x'])):
        s.id2group_label[counter] = [i, j]
        counter += 1

# Set the figure size (adjust the width and height as needed)
fig = plt.figure(figsize=(12, 6))

# Set the size and position of the axis within the figure
ax = plt.axes((0.01, 0.1, 0.7, 0.8))  # [left, bottom, width, height]

# Create a scatter plot for each class with different marker types
for class1 in class1_list:
    if 'large' in class1['label']:
        s_ = class1['volume'] / 5.
    else:
        s_ = class1['volume']
    plt.scatter(class1['x'], class1['y'], cmap='viridis', edgecolors='k', s=s_, marker=class1['marker'],
                             label=class1['label'])
for class2 in class2_list:
    if 'large' in class2['label']:
        s_ = class2['volume'] / 5.
    else:
        s_ = class2['volume'] * 5
    plt.scatter(class2['x'], class2['y'], cmap='viridis', edgecolors='k', s=s_, marker=class2['marker'],
                             label=class2['label'])

plt.legend()

# Add colorbar to show the mapping of values to colors
cbar = plt.colorbar()
cbar.set_label('Z values')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot with Color based on Z')


# Enable hovering with mplcursors
def cursor_hover(sel):
    class_type = sel.annotation.get_text().split('\n')[0]
    if class_type == 'RTF: small':
        cur_data = class1_list[0]
        nn_start_ind = 1
        nn_end_ind = None
    elif class_type == 'RTF: large':
        cur_data = class1_list[1]
        nn_start_ind = 1
        nn_end_ind = None
    elif class_type == 'RTF: hall':
        cur_data = class1_list[2]
        nn_start_ind = 1
        nn_end_ind = None
    elif class_type == 'Audio: hall':
        cur_data = class2_list[s.t2id[class_type.split(' ')[1]]]
        nn_start_ind = 0
        nn_end_ind = -1
    elif class_type == 'Audio: small':
        cur_data = class2_list[s.t2id[class_type.split(' ')[1]]]
        nn_start_ind = 0
        nn_end_ind = -1
    elif class_type == 'Audio: large':
        cur_data = class2_list[s.t2id[class_type.split(' ')[1]]]
        nn_start_ind = 0
        nn_end_ind = -1
    elif class_type == 'Audio: small_shelter':
        cur_data = class2_list[s.t2id[class_type.split(' ')[1]]]
        nn_start_ind = 0
        nn_end_ind = -1
    else:
        assert 0
    assert cur_data['label'] == class_type
    w_clicked = cur_data['volume'][sel.index]
    x_clicked = cur_data['x'][sel.index]
    y_clicked = cur_data['y'][sel.index]

    sel.annotation.set_text(f"Class: {class_type}" + '\n' +
                            f"Volume: {w_clicked:.2f}" + '\n' +
                            f"X: {sel.target_.data[0]:.2f}" + '\n' +
                            f"Y: {sel.target_.data[1]:.2f}")

    # Find 5 nearest neighbors
    all_dists = []

    for class_ in class1_list:
        if s.tsne_dist:
            # calc distance in the TSNE projected ones.
            distances = np.sqrt((class_['x'] - x_clicked)**2 + (class_['y'] - y_clicked)**2)
        else:
            # calc distance in d-dimensional embedings

            # cur_norm = np.linalg.norm(cur_data['embed'][sel.index]).reshape(-1, 1)
            cur_emb = cur_data['embed'][sel.index].reshape(1, -1) # / cur_norm

            # class_norm = np.linalg.norm(class_['embed'], axis=1).reshape(-1, 1)
            class_emb = class_['embed'] # / class_norm
            distances = np.dot(class_emb, cur_emb.T).flatten()

            # for competability with the TSNE features (where we calc the argmin), we multiply it by -1
            distances = -distances

        all_dists.append(distances)
    all_dists = np.hstack(all_dists)
    class1_nearest_indices = np.argsort(all_dists)[:s.n_neighbours + 1]

    class_1_type_and_ind = [s.id2group_label[x] for x in class1_nearest_indices]

    if class_type.startswith('RTF'):
        assert class_1_type_and_ind[0][1] == sel.index
    class_1_type_and_ind = class_1_type_and_ind[nn_start_ind:nn_end_ind]
    s.cur_samples = class_1_type_and_ind

    if s.classify:
        if len(s.scatter) > 0:
            for a in s.scatter:
                a.remove()
            s.scatter = []
        assert len(s.scatter) == 0
        for cls_type, cls_id in s.cur_samples:
            a = plt.scatter(class1_list[cls_type]['x'][cls_id],
                            class1_list[cls_type]['y'][cls_id],
                            cmap='viridis', edgecolors='k',
                            c='k',
                            s=class1_list[cls_type]['volume'][cls_id] * 5,
                            marker=class1_list[cls_type]['marker'])
            # from IPython import embed; embed()
            s.scatter.append(a)
        s.prev_samples = s.cur_samples
        fig.show()


    headers = ['room_type', 'index', 'volume', 'dim (x,y,z)']
    str_list = []
    for cls_type, cls_id in class_1_type_and_ind:
        # from IPython import embed; embed()
        str_list.append([f'{class1_list[cls_type]["label"].replace(":", "")}',
                         f'{cls_id:05d}',
                         f'{class1_list[cls_type]["volume"][cls_id]:0.3f}',
                         f'{class1_list[cls_type]["room_type"][cls_id]}'])

    md_str = tabulate(str_list,
                      headers=headers,
                      )
    if s.text is not None:
        s.text.remove()
    s.text = plt.text(1.3, 0.5, f'NN data based on {"TSNE" if s.tsne_dist else "Cosine sim"} for {class_type} indx {sel.index}'
                      + '\n' + md_str, rotation=0, va='center', ha='left', transform=plt.gca().transAxes)

def show_popup_window():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    result = simpledialog.askstring('Nearest Neighbour setup', f"Enter number:")
    if result:
        messagebox.showinfo("Set NN to: ", result)
        s.n_neighbours = int(result)
    root.destroy()  # Destroy the hidden main window when done

def on_key(event):
    if event.key == 'h':
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Help", "Key options: \nt: calc disatnces w/wo TSNE\nn: set n NN\nc: Classify")
        root.destroy()  # Destroy the hidden main window when done

    if event.key == 't':
        s.tsne_dist = not s.tsne_dist
    if event.key == 'n':
        show_popup_window()
    elif event.key == 'c':
        s.classify = not s.classify
        if not s.classify and len(s.scatter) > 0:
            for a in s.scatter:
                a.remove()
            s.scatter = []
    # if s.scatter is not None:
        #     for a in s.scatter:
        #         a.remove()
        #
        # s.scatter = []
        # for cls_type, cls_id in s.cur_samples:
        #     a = plt.scatter(class1_list[cls_type]['x'][cls_id],
        #                     class1_list[cls_type]['y'][cls_id],
        #                     cmap='viridis', edgecolors='k',
        #                     c='k',
        #                     s=class1_list[cls_type]['volume'][cls_id] * 5,
        #                     marker=class1_list[cls_type]['marker'])
        #     # from IPython import embed; embed()
        #     s.scatter.append(a)
        # s.prev_samples = s.cur_samples
        # fig.show()

mplcursors.cursor(hover=True, highlight=True).connect("add", cursor_hover)
fig.canvas.mpl_connect('key_press_event', on_key)
# Show the plot
plt.show()