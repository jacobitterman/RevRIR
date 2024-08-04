import os
import pandas as pd
import numpy as np
import pickle as pkl

from ..utils import HOME, get_audio
ACE_BM_PATH = f'/{HOME}/asr/scratch/carir/benchmarks/room_classifier/ace/ACE/Algorithms/RIR_Estimation_and_Decimation/V1/IRestim'


def calc_room_type(room_dims, print_room_dims=False):
    small = [(1.5, 4), (2.5, 4.5), (2, 3.5)]
    large = [(6, 13), (6, 12), (2, 5)]
    hall = [(1, 3), (7, 13), (2, 3.5)]

    if print_room_dims:
        print(f'small: {np.prod([x[0] for x in small]):0.3f}-{np.prod([x[1] for x in small]):0.3f}')
        print(f'large: {np.prod([x[0] for x in large]):0.3f}-{np.prod([x[1] for x in large]):0.3f}')
        print(f'hall: {np.prod([x[0] for x in hall]):0.3f}-{np.prod([x[1] for x in hall]):0.3f}')

    x, y, z = room_dims
    if small[0][0] <= x <= small[0][1] and small[1][0] <= y <= small[1][1] and small[2][0] <= z <= small[2][1]:
        return 'small'
    elif large[0][0] <= x <= large[0][1] and large[1][0] <= y <= large[1][1] and large[2][0] <= z <= large[2][1]:
        return 'large'
    elif hall[0][0] <= x <= hall[0][1] and hall[1][0] <= y <= hall[1][1] and hall[2][0] <= z <= hall[2][1]:
        return 'hall'
    else:
        print(f'unknown type: V={x * y * z:0.3f} m^3')
        return 'unknown'


def parse_data(fs=8000, verbose=False):
    from ..utils import HOME
    save_dir = f"/{HOME}/asr/scratch/carir/benchmarks/room_classifier/ace/extracted_rirs_normalized"
    wav_name_2_paper_name = {'Mobile_508_1_RIR_8k.wav': ['Lecture_Room_1_m1', 10.3, 9.07, 2.63, 0.371, 'large'],
                             'Mobile_508_2_RIR_8k.wav': ['Lecture_Room_1_m2', 10.3, 9.07, 2.63, 0.371, 'large'],
                             'Mobile_403a_1_RIR_8k.wav': ['Lecture_Room_2_m1', 3.32, 4.83, 2.95, 0.332, 'small'],
                             'Mobile_403a_2_RIR_8k.wav': ['Lecture_Room_2_m2', 3.32, 4.83, 2.95, 0.332, 'small'],
                             'Mobile_503_1_RIR_8k.wav': ['Meeting_Room_1_m1', 6.61, 5.11, 2.95, 0.437, 'large'],
                             'Mobile_503_2_RIR_8k.wav': ['Meeting_Room_1_m2', 6.61, 5.11, 2.95, 0.437, 'large'],
                             'Mobile_611_1_RIR_8k.wav': ['Meeting_Room_2_m1', 6.93, 9.73, 3.0, 0.638, 'large'],
                             'Mobile_611_2_RIR_8k.wav': ['Meeting_Room_2_m2', 6.93, 9.73, 3.0, 0.638, 'large'],
                             'Mobile_502_1_RIR_8k.wav': ['Office_1_m1', 3.22, 5.1, 2.94, 0.39, 'small'],
                             'Mobile_502_2_RIR_8k.wav': ['Office_1_m2', 3.22, 5.1, 2.94, 0.39, 'small'],
                             'Mobile_803_1_RIR_8k.wav': ['Office_2_m1', 13.6, 9.29, 2.94, 1.22, 'large'],
                             'Mobile_803_2_RIR_8k.wav': ['Office_2_m2', 13.6, 9.29, 2.94, 1.22, 'large'],
                             'Mobile_EE_lobby_1_RIR_8k.wav': ['Building_Lobby_1_m1', 4.47, 5.13, 3.18, 0.646, 'large'],
                             'Mobile_EE_lobby_2_RIR_8k.wav': ['Building_Lobby_1_m2', 4.47, 5.13, 3.18, 0.646, 'large']}

    for desc in ["large", "small"]:
        d_list = []
        rirs = []
        for k, v in wav_name_2_paper_name.items():
            if v[5] != desc:
                continue
            wav_path = os.path.join(ACE_BM_PATH, k)
            rir = get_audio(wav_path, fs=fs)
            from ..utils import rms
            r = rms(rir[:int(fs * v[4] * 1.2)])
            rir = rir / r / 500
            rirs.append(rir)

            d = {'c': -1,  # Sound velocity (m/s)
                 'fs': fs,  # Sample frequency (samples/s)
                 'r': [0, 0, 0],  # Receiver position(s) [x y z] (m)
                 's': [0, 0, 0],  # Source position [x y z] (m)
                 'L': (v[1], v[2], v[3]),  # Room dimensions before randomization [x y z] (m)
                 'L_new': (v[1], v[2], v[3]),  # Actuall Room dimensions [x y z] (m)
                 'reverberation_time': v[4],  # Reverberation time (s)
                 'beta': [0, 0, 0, 0, 0, 0],
                 'nsample': None,  # Number of output samples
                 'wav_path': wav_path,
                 'paper_name': v[0],
                 'room_type': v[5],  # calc_room_type([v[1], v[2], v[3]])
                 }
            d_list.append(d)
        df = pd.DataFrame(d_list)
        if verbose:
            print(df)
        df.to_csv(os.path.join(save_dir, f'rirs_{desc}_md.csv'))
        max_size = np.max([x.size for x in rirs])
        pkl.dump(("ace", max_size, rirs), open(os.path.join(save_dir, f'rirs_{desc}.pkl'), 'wb'), protocol=-1)


def main():
    df = parse_data(verbose=True)


if __name__ == '__main__':
    main()
