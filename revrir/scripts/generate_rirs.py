import os.path
from typing import Optional, List, Union, Callable
from time import sleep
from copy import copy
import psutil
from tqdm import tqdm
try:
    from p_tqdm import p_map
except ImportError:
    print("pip install p_tqdm")
    exit(0)

import numpy as np
import rir_generator as rir
import pandas as pd
import pickle as pkl

from ..utils import makedirs_p, rms
from .room_params import extractFreqParams, extractTimeParams, parse_features

def single_par_map(func: Callable,
                   params: Union[List[dict], List],
                   fixed_params: Optional[dict]=None,
                   limit_num_cpus: bool=True,
                   run_locally: bool = False):
    """
    :param run_locally: boolean parameter (one vs. many cores)
    :param func: applied function
    :param params: varying parameter(s) as either a list of dictionaries or a list. In the latter case, it's assumed
    the (single) varying parameter corresponding to params is the first positional parameter of the function func
    :param fixed_params: fixed parameters as a dictionary
    """
    fixed_params_ = {} if not fixed_params else fixed_params
    if type(params[0]) is dict:
        partial_fun = lambda x: func(**x, **fixed_params_)
    else:
        partial_fun = lambda x: func(x, **fixed_params_)
    if run_locally:
        res = []
        # for par in tqdm(params, total=len(params)):
        #     res.append(partial_fun(par))
        res = [partial_fun(par) for par in tqdm(params, total=len(params))]
        return res
    else:
        num_cpus = psutil.cpu_count(logical=False)/2 if limit_num_cpus else psutil.cpu_count(logical=False)
        return p_map(partial_fun, params, num_cpus = 15)
"""
Example
"""

def my_pow(base, exp):
    sleep(1)
    return base ** exp

def example_run():
    # using my_pow to compute squares
    _fixed_params = {'exp': 2}
    _params_list = [{'base': i} for i in range(100)]
    # the following also works in this case since base is the first parameter of my_pow
    # _params_list = range(100)
    _res = single_par_map(my_pow, params= _params_list, fixed_params=_fixed_params)
    print(_res)


def generate_n_rirs(n_samples_to_generate,
                    L=[3, 4, 3],
                    fs=8000,
                    c=340,
                    reverberation_time=[0.4, 1.0],
                    desc='small',
                    save_dir='.'):
    nsample = int((reverberation_time[1] + 0.3) * fs)
    md_dict_list = []
    for _ in tqdm(range(n_samples_to_generate), desc='preparing meta-data'):
        s = np.hstack((np.random.uniform(0, L[0]),
                       np.random.uniform(0, L[1]),
                       np.random.uniform(0, L[2]),
                       ))
        r = np.vstack((np.random.uniform(0, L[0]),
                       np.random.uniform(0, L[1]),
                       np.random.uniform(0, L[2]),
                       )).T
        t60 = np.random.uniform(reverberation_time[0], reverberation_time[1])
        md_dict = {'c': c,  # Sound velocity (m/s)
                   'fs': fs,  # Sample frequency (samples/s)
                   'r': r,  # Receiver position(s) [x y z] (m)
                   's': s,  # Source position [x y z] (m)
                   'L': L,  # Room dimensions [x y z] (m)
                   'reverberation_time': t60,  # Reverberation time (s)
                   'nsample': nsample,  # Number of output samples
                   }
        md_dict_list.append(md_dict)
    df = pd.DataFrame(md_dict_list)
    df.to_csv(os.path.join(save_dir, f'rirs_{desc}_md.csv'))
    _res = single_par_map(rir.generate, params=md_dict_list, fixed_params=None)
    rirs = np.array(_res)[:, :, 0].astype(np.float32)
    pkl.dump(rirs, open(os.path.join(save_dir, f'rirs_{desc}.pkl'), 'wb'), protocol=-1)
    print(f'the data size is {len(rirs.tobytes())/1024./1024.:0.2f} MBs')
    print('finished')


def calc_rt60(r, L, s, c, beta, **kwargs):
    # taken from rir.generate code
    r = np.atleast_2d(np.asarray(r, dtype=np.double)).T.copy()
    assert r.shape[0] == 3

    L = np.asarray(L, dtype=np.double)
    assert L.shape == (3,)

    s = np.asarray(s, dtype=np.double)
    assert s.shape == (3,)

    if beta is not None:
        beta = np.asarray(beta, dtype=np.double)
        assert beta.shape == (6,) or beta.shape == (3, 2)
        beta = beta.reshape(3, 2)

    if (r > L[:, None]).any() or (r < 0).any():
        raise ValueError("r is outside the room")

    if (s > L).any() or (s < 0).any():
        raise ValueError("s is outside the room")

    # Volume of room
    V = np.prod(L)
    # Surface area of walls
    A = L[::-1] * np.roll(L[::-1], 1)

    alpha = np.sum(np.sum(1 - beta ** 2, axis=1) * np.sum(A))

    reverberation_time = max(
        24 * np.log(10.0) * V / (c * alpha),
        0.128,
    )
    return reverberation_time


def generate_n_rirs_v2(n_samples_to_generate,
                       L=[3, 4, 3],
                       fs=8000,
                       c=343,
                       reverberation_time=None,  #[0.4, 1.0],
                       beta=[0.8, 0.95],
                       desc='small',
                       save_dir='.'):
    makedirs_p(save_dir)

    md_dict_list = []

    for _ in tqdm(range(n_samples_to_generate), desc='preparing meta-data'):
        s = np.hstack((np.random.uniform(0.5, L[0] - 0.5),
                       np.random.uniform(0.5, L[1] - 0.5),
                       np.random.uniform(0.5, L[2] - 0.5),
                       ))
        r = np.vstack((np.random.uniform(0.5, L[0] - 0.5),
                       np.random.uniform(0.5, L[1] - 0.5),
                       np.random.uniform(0.5, L[2] - 0.5),
                       )).T
        beta_inst = np.random.uniform(beta[0], beta[1])
        md_dict = {'c': c,  # Sound velocity (m/s)
                   'fs': fs,  # Sample frequency (samples/s)
                   'r': r,  # Receiver position(s) [x y z] (m)
                   's': s,  # Source position [x y z] (m)
                   'L': L,  # Room dimensions [x y z] (m)
                   'reverberation_time': reverberation_time,  # Reverberation time (s)
                   'beta': [beta_inst] * 6,
                   'nsample': None,  # Number of output samples
                   }
        if reverberation_time is None:
            rt60 = calc_rt60(**md_dict)
            md_dict['nsample'] = int(rt60 * fs * 1.2)
        md_dict_list.append(md_dict)
    df = pd.DataFrame(md_dict_list)
    df.to_csv(os.path.join(save_dir, f'rirs_{desc}_md.csv'))
    _res = single_par_map(rir.generate, params=md_dict_list, fixed_params=None)
    rirs = [x[:, 0].astype(np.float32) for x in _res]
    max_size = np.max([x.size for x in rirs])
    pkl.dump(("v2", max_size, rirs), open(os.path.join(save_dir, f'rirs_{desc}.pkl'), 'wb'), protocol=-1)
    print(f'the data size is {np.sum([len(x.tobytes()) for x in rirs])/1024./1024.:0.2f} MBs')
    print(f'max size: {max_size}')
    print('finished')

def rir_generate_wrapper(verbose = False, normalize_rms = False, **md_dict):
    md_dict_to_rir_gen = copy(md_dict)
    if 'L_new' in md_dict:
        md_dict_to_rir_gen['L'] = md_dict['L_new']
        del md_dict_to_rir_gen['L_new']
    if 'room_idx' in md_dict:
        del md_dict_to_rir_gen['room_idx']
    out = rir.generate(**md_dict_to_rir_gen)[:, 0].astype(np.float32)

    if normalize_rms:  # normalize_rms = True means version v3_1 / v3_3
        out_ = out
        out_rms = rms(out_)
        out = out_ / out_rms / 100.
        if verbose:
            # print(rms(out))
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.plot(out_)
            plt.subplot(212)
            plt.plot(out)
            plt.show()
    return out

def rir_generate_wrapper_with_feature_vector(verbose = False, normalize_rms = False, **md_dict):
    md_dict_to_rir_gen = copy(md_dict)
    if 'L_new' in md_dict:
        md_dict_to_rir_gen['L'] = md_dict['L_new']
        del md_dict_to_rir_gen['L_new']
    if 'room_idx' in md_dict:
        del md_dict_to_rir_gen['room_idx']
    h = rir.generate(**md_dict_to_rir_gen)

    timeParams = extractTimeParams(h,
                                   md_dict['fs'],
                                   startRT=0.45,
                                   finishRT=0.75,
                                   Nstart=0.025,
                                   Nfinish=0.05,
                                   T30flag=False,
                                   rmsFlag=False,
                                   std_thresh=1.5,
                                   plots=verbose)
    freqParams = extractFreqParams(h,
                                   md_dict['fs'],
                                   refFlag=False,
                                   refDet=0,
                                   std_thresh=1.5,
                                   f0=100,
                                   Kbands=2,
                                   Nbands=5,
                                   plots=verbose)
    features_vec, features_name = parse_features(timeParams, freqParams)
    features_vec = np.array(features_vec, dtype=np.float32)
    if np.any(np.isnan(features_vec)):
        features_vec[np.where(np.isnan(features_vec))[0]] = 0
        # import pickle as pkl
        # pkl.dump((md_dict, h, features_vec, features_name, timeParams, freqParams), open('/home/talr/tmp/dbg.pkl', 'wb'))
        # from IPython import embed; embed()
    out = h[:, 0].astype(np.float32)
    if normalize_rms:  # normalize_rms = True means version v3_1 / v3_3
        out_ = out
        out_rms = rms(out_)
        out = out_ / out_rms / 100.
        if verbose:
            # print(rms(out))
            import matplotlib.pyplot as plt
            plt.subplot(211)
            plt.plot(out_)
            plt.subplot(212)
            plt.plot(out)
            plt.show()
    return out, features_vec, features_name

def generate_n_rirs_v3(n_samples_to_generate,
                       run_locally=False,
                       L=[3, 4, 3],
                       wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       source_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       minimal_r_s_dist = 0.5,
                       beta=[0.88, 0.9],
                       fs=8000,
                       c=343,
                       reverberation_time=None,  #[0.4, 1.0],
                       normalize_rms = False,
                       desc='small',
                       save_dir='.',
                       version="v3",
                       verbose = False):
    assert version in ["v3", "v3.1", "v3.2", "v3.3", "tmp", "v3.4", "v3.6"], version
    makedirs_p(save_dir)

    md_dict_list = []

    for _ in tqdm(range(n_samples_to_generate), desc='preparing meta-data'):
        new_L = np.hstack((np.random.uniform(max(1.05, L[0] - wall_margin[0][0]), L[0] + wall_margin[0][1]),
                           np.random.uniform(max(1.05, L[1] - wall_margin[1][0]), L[1] + wall_margin[1][1]),
                           np.random.uniform(max(1.05, L[2] - wall_margin[2][0]), L[2] + wall_margin[2][1]),
                           ))

        s = np.hstack((np.random.uniform(source_from_wall_margin[0][0], new_L[0] - source_from_wall_margin[0][1]),
                       np.random.uniform(source_from_wall_margin[1][0], new_L[1] - source_from_wall_margin[1][1]),
                       np.random.uniform(source_from_wall_margin[2][0], new_L[2] - source_from_wall_margin[2][1]),
                       ))

        d = minimal_r_s_dist - 1  # force first run
        i = 0
        MAX_ATTEMPTS = 100
        while d < minimal_r_s_dist and i < MAX_ATTEMPTS:
            r = np.vstack(
                (np.random.uniform(receiver_from_wall_margin[0][0], new_L[0] - receiver_from_wall_margin[0][1]),
                 np.random.uniform(receiver_from_wall_margin[1][0], new_L[1] - receiver_from_wall_margin[1][1]),
                 np.random.uniform(receiver_from_wall_margin[2][0], new_L[2] - receiver_from_wall_margin[2][1]),
                 )).T
            d = np.linalg.norm(r[0]-s)
            i = i + 1
        if i >= MAX_ATTEMPTS:
            raise Exception()

        beta_gap = min(beta[1] - beta[0] - 0.05, 0.15)
        beta_baseline = np.random.uniform(beta[0], beta[1] - (beta_gap / 2))
        beta_inst = [min(beta[1], beta_baseline + np.random.uniform(0, beta_gap)) for i in range(6)]

        md_dict = {'c': c,  # Sound velocity (m/s)
                   'fs': fs,  # Sample frequency (samples/s)
                   'r': r,  # Receiver position(s) [x y z] (m)
                   's': s,  # Source position [x y z] (m)
                   'L': L,  # Room dimensions before randomization [x y z] (m)
                   'L_new': new_L, # Actuall Room dimensions [x y z] (m)
                   'reverberation_time': reverberation_time,  # Reverberation time (s)
                   'beta': beta_inst,
                   'nsample': None,  # Number of output samples
                   'verbose': verbose,
                   'normalize_rms':  normalize_rms,
                   }
        if reverberation_time is None:
            md_dict["L"] = md_dict["L_new"]
            rt60 = calc_rt60(**md_dict)
            md_dict["L"] = L

            md_dict['nsample'] = int(rt60 * fs * 1.2)
        md_dict_list.append(md_dict)
    df = pd.DataFrame(md_dict_list)
    df.to_csv(os.path.join(save_dir, f'rirs_{desc}_md.csv'))

    rirs = single_par_map(rir_generate_wrapper, params=md_dict_list, fixed_params=None, run_locally=run_locally)

    if verbose:
        import scipy.signal as ss
        import librosa
        from ..utils import get_audio_paths, play_audio_ffplay
        a, _ = librosa.load(get_audio_paths("test")[0], sr=8000)
        signal = ss.convolve(rirs[0], a)
        play_audio_ffplay(signal, fs)
        import matplotlib.pyplot as plt
        plt.plot([len(np.unique(x)) for x in rirs]); plt.show()
        from IPython import embed; embed()
    max_size = np.max([x.size for x in rirs])
    pkl.dump((version, max_size, rirs), open(os.path.join(save_dir, f'rirs_{desc}.pkl'), 'wb'), protocol=-1)
    print(f'the data size is {np.sum([len(x.tobytes()) for x in rirs])/1024./1024.:0.2f} MBs')
    print(f'max size: {max_size}')
    print('finished')


def generate_n_rirs_v4(n_samples_per_room,
                       run_locally=False,
                       L = ([1.5, 2.5, 2.5], [4, 4.5, 3]),
                       wall_margin = [0.5, 0.5, 0.5],
                       source_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       minimal_r_s_dist = 0.5,
                       beta=[0.88, 0.9],
                       fs=8000,
                       c=343,
                       reverberation_time=None,  #[0.4, 1.0],
                       normalize_rms = False,
                       desc='small',
                       save_dir='.',
                       version="v4",
                       verbose = False):
    assert version in ["v4", "v4.1"], version
    makedirs_p(save_dir)

    md_dict_list = []

    L_min, L_max = L
    rooms = []
    for l1 in np.arange(L_min[0], L_max[0] + 0.1, wall_margin[0]):
        for l2 in np.arange(L_min[1], L_max[1] + 0.1, wall_margin[1]):
            for l3 in np.arange(L_min[2], L_max[2] + 0.1, wall_margin[2]):
                if (l2, l1, l3) not in rooms:
                    rooms.append((l1, l2, l3))

    print(f"created {len(rooms)} rooms. will create {n_samples_per_room} rirs per room")

    for idx, room in enumerate(rooms):
        for _ in tqdm(range(n_samples_per_room), desc=f'preparing meta-data for room {idx}'):
            s = np.hstack((np.random.uniform(source_from_wall_margin[0][0], room[0] - source_from_wall_margin[0][1]),
                           np.random.uniform(source_from_wall_margin[1][0], room[1] - source_from_wall_margin[1][1]),
                           np.random.uniform(source_from_wall_margin[2][0], room[2] - source_from_wall_margin[2][1]),
                           ))

            d = minimal_r_s_dist - 1  # force first run
            i = 0
            MAX_ATTEMPTS = 100
            while d < minimal_r_s_dist and i < MAX_ATTEMPTS:
                r = np.vstack(
                    (np.random.uniform(receiver_from_wall_margin[0][0], room[0] - receiver_from_wall_margin[0][1]),
                     np.random.uniform(receiver_from_wall_margin[1][0], room[1] - receiver_from_wall_margin[1][1]),
                     np.random.uniform(receiver_from_wall_margin[2][0], room[2] - receiver_from_wall_margin[2][1]),
                     )).T
                d = np.linalg.norm(r[0]-s)
                i = i + 1
            if i >= MAX_ATTEMPTS:
                raise Exception()

            if isinstance(beta, float):
                beta_inst = [beta] * 6
            elif beta[1] - beta[0] > 0.05:
                beta_gap = min(beta[1] - beta[0] - 0.05, 0.15)
                beta_baseline = np.random.uniform(beta[0], beta[1] - (beta_gap / 2))
                beta_inst = [min(beta[1], beta_baseline + np.random.uniform(0, beta_gap)) for i in range(6)]
            else:
                beta_inst = [np.random.uniform(beta[0], beta[1]) for i in range(6)]

            md_dict = {'c': c,  # Sound velocity (m/s)
                       'fs': fs,  # Sample frequency (samples/s)
                       'r': r,  # Receiver position(s) [x y z] (m)
                       's': s,  # Source position [x y z] (m)
                       'L': room,  # Room dimensions before randomization [x y z] (m)
                       'room_idx': idx,
                       'reverberation_time': reverberation_time,  # Reverberation time (s)
                       'beta': beta_inst,
                       'nsample': None,  # Number of output samples
                       'verbose': verbose,
                       'normalize_rms':  normalize_rms,
                       }
            if reverberation_time is None:
                rt60 = calc_rt60(**md_dict)

            md_dict['nsample'] = int(rt60 * fs * 1.2)
            md_dict_list.append(md_dict)

    df = pd.DataFrame(md_dict_list)
    df.to_csv(os.path.join(save_dir, f'rirs_{desc}_md.csv'))

    rirs = single_par_map(rir_generate_wrapper, params=md_dict_list, fixed_params=None, run_locally=run_locally)

    if verbose:
        import scipy.signal as ss
        import librosa
        from ..utils import get_audio_paths, play_audio_ffplay
        a, _ = librosa.load(get_audio_paths("test")[0], sr=8000)
        signal = ss.convolve(rirs[0], a)
        play_audio_ffplay(signal, fs)
        import matplotlib.pyplot as plt
        plt.plot([len(np.unique(x)) for x in rirs]); plt.show()
        from IPython import embed; embed()
    max_size = np.max([x.size for x in rirs])
    pkl.dump((version, max_size, rooms, n_samples_per_room, rirs), open(os.path.join(save_dir, f'rirs_{desc}.pkl'), 'wb'), protocol=-1)
    print(f'the data size is {np.sum([len(x.tobytes()) for x in rirs])/1024./1024.:0.2f} MBs')
    print(f'max size: {max_size}')
    print('finished')



def generate_n_rirs_v5(n_samples_per_room,
                       run_locally=False,
                       L = ([1.5, 2.5, 2.5], [4, 4.5, 3]),
                       wall_margin = [0.5, 0.5, 0.5],
                       source_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                       minimal_r_s_dist = 0.5,
                       beta=[0.88, 0.9],
                       fs=8000,
                       c=343,
                       reverberation_time=None,  #[0.4, 1.0],
                       normalize_rms = False,
                       desc='small',
                       save_dir='.',
                       version="v5",
                       verbose = False):
    assert version in ["v5"], version

    makedirs_p(save_dir)

    md_dict_list = []

    L_min, L_max = L
    rooms = []
    for l1 in np.arange(L_min[0], L_max[0] + 0.1, wall_margin[0]):
        for l2 in np.arange(L_min[1], L_max[1] + 0.1, wall_margin[1]):
            for l3 in np.arange(L_min[2], L_max[2] + 0.1, wall_margin[2]):
                if (l2, l1, l3) not in rooms:
                    rooms.append((l1, l2, l3))

    print(f"created {len(rooms)} rooms. will create {n_samples_per_room} rirs per room")

    for idx, room in enumerate(rooms):
        for _ in tqdm(range(n_samples_per_room), desc=f'preparing meta-data for room {idx}'):
            s = np.hstack((np.random.uniform(source_from_wall_margin[0][0], room[0] - source_from_wall_margin[0][1]),
                           np.random.uniform(source_from_wall_margin[1][0], room[1] - source_from_wall_margin[1][1]),
                           np.random.uniform(source_from_wall_margin[2][0], room[2] - source_from_wall_margin[2][1]),
                           ))

            d = minimal_r_s_dist - 1  # force first run
            i = 0
            MAX_ATTEMPTS = 100
            while d < minimal_r_s_dist and i < MAX_ATTEMPTS:
                r = np.vstack(
                    (np.random.uniform(receiver_from_wall_margin[0][0], room[0] - receiver_from_wall_margin[0][1]),
                     np.random.uniform(receiver_from_wall_margin[1][0], room[1] - receiver_from_wall_margin[1][1]),
                     np.random.uniform(receiver_from_wall_margin[2][0], room[2] - receiver_from_wall_margin[2][1]),
                     )).T
                d = np.linalg.norm(r[0]-s)
                i = i + 1
            if i >= MAX_ATTEMPTS:
                raise Exception()

            if isinstance(beta, float):
                beta_inst = [beta] * 6
            elif beta[1] - beta[0] > 0.05:
                beta_gap = min(beta[1] - beta[0] - 0.05, 0.15)
                beta_baseline = np.random.uniform(beta[0], beta[1] - (beta_gap / 2))
                beta_inst = [min(beta[1], beta_baseline + np.random.uniform(0, beta_gap)) for i in range(6)]
            else:
                beta_inst = [np.random.uniform(beta[0], beta[1]) for i in range(6)]

            md_dict = {'c': c,  # Sound velocity (m/s)
                       'fs': fs,  # Sample frequency (samples/s)
                       'r': r,  # Receiver position(s) [x y z] (m)
                       's': s,  # Source position [x y z] (m)
                       'L': room,  # Room dimensions before randomization [x y z] (m)
                       'room_idx': idx,
                       'reverberation_time': reverberation_time,  # Reverberation time (s)
                       'beta': beta_inst,
                       'nsample': None,  # Number of output samples
                       'verbose': verbose,
                       'normalize_rms':  normalize_rms,
                       }
            if reverberation_time is None:
                rt60 = calc_rt60(**md_dict)

            md_dict['nsample'] = int(rt60 * fs * 1.2)
            md_dict_list.append(md_dict)

    rirs_new = single_par_map(rir_generate_wrapper_with_feature_vector, params=md_dict_list, fixed_params=None, run_locally=run_locally)
    rirs, rirs_features_vecs, rirs_features_names = zip(*rirs_new)
    assert all([rirs_features_names[0] == r for r in rirs_features_names])

    assert len(md_dict_list) == len(rirs_features_vecs) == len(rirs_features_names)
    final_md_dict_list = []
    for md_dict, f_v, f_n in zip(md_dict_list, rirs_features_vecs, rirs_features_names):
        md_dict_final = copy(md_dict)
        for k, v in zip(f_n, f_v):
            md_dict_final[k] = v
        final_md_dict_list.append(md_dict_final)
    df = pd.DataFrame(final_md_dict_list)
    df.to_csv(os.path.join(save_dir, f'rirs_{desc}_md.csv'))

    if verbose:
        import matplotlib.pyplot as plt
        import scipy.signal as ss
        import librosa
        from ..utils import get_audio_paths, play_audio_ffplay

        plt.imshow(np.vstack(np.log(x[1]) for x in rirs_new).T, aspect='auto', interpolation='none')
        plt.yticks(range(40), labels=rirs_new[0][2])
        plt.colorbar()
        plt.show()

        a, _ = librosa.load(get_audio_paths("test")[0], sr=8000)
        signal = ss.convolve(rirs[0], a)
        play_audio_ffplay(signal, fs)

        plt.plot([len(np.unique(x)) for x in rirs]); plt.show()

    max_size = np.max([x.size for x in rirs])
    pkl.dump((version, max_size, rooms, n_samples_per_room, rirs_features_vecs, rirs), open(os.path.join(save_dir, f'rirs_{desc}.pkl'), 'wb'), protocol=-1)

    print(f'the data size is {np.sum([len(x.tobytes()) for x in rirs])/1024./1024.:0.2f} MBs')
    print(f'max size: {max_size}')
    print('finished writing pkl')

    write_binaries = False
    if write_binaries:
        from ..utils import IndexedBinary, encode_buff
        with IndexedBinary(os.path.join(save_dir, f'rirs_{desc}.bin'), 'w') as bin:
            for rir, fv in zip(rirs, rirs_features_vecs):
                buff = encode_buff({'rir': rir,
                                    'feature_vec': fv})
                bin.write(buff)
        print('finished writing bin')


def dbg(verbose=True):
    import pickle as pkl
    md_dict, h, features_vec, features_name, timeParams, freqParams = pkl.load(open('/home/talr/tmp/dbg.pkl', 'rb'))

    timeParams_ = extractTimeParams(h,
                                   md_dict['fs'],
                                   startRT=0.45,
                                   finishRT=0.75,
                                   Nstart=0.025,
                                   Nfinish=0.05,
                                   T30flag=False,
                                   rmsFlag=False,
                                   std_thresh=1.5,
                                   plots=verbose)
    freqParams_ = extractFreqParams(h,
                                   md_dict['fs'],
                                   refFlag=False,
                                   refDet=0,
                                   std_thresh=1.5,
                                   f0=100,
                                   Kbands=2,
                                   Nbands=5,
                                   plots=verbose)
    print('dbg passes')
    # from IPython import embed; embed()


def visualize_features():
    import pickle as pkl
    import matplotlib.pyplot as plt
    import numpy as np

    save_dir = '/home/talr/tmp/rir/v5/rirs_v5_benchmark_3K'
    small = pkl.load(open(os.path.join(save_dir, 'rirs_small.pkl'), 'rb'))
    small_concat = np.vstack(small[5])

    large = pkl.load(open(os.path.join(save_dir, 'rirs_large.pkl'), 'rb'))
    large_concat = np.vstack(large[5])

    hall = pkl.load(open(os.path.join(save_dir, 'rirs_hall.pkl'), 'rb'))
    hall_concat = np.vstack(hall[5])

    plt.imshow(np.vstack([small_concat, large_concat, hall_concat]).T, aspect='auto', interpolation='none')
    # plt.yticks(range(40), labels=rirs_new[0][2])
    plt.colorbar()
    plt.show()


if __name__=="__main__":
    from ..utils import HOME
    # example_run()
    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_train_30K'
    #
    # samples_per_room = 100000
    # generate_n_rirs_v2(samples_per_room, L=[3, 4, 3], desc='small', save_dir=save_dir_)
    # generate_n_rirs_v2(samples_per_room, L=[4, 7, 3.5], desc='large', save_dir=save_dir_)
    # generate_n_rirs_v2(samples_per_room, L=[2, 6, 2.5], desc='hall', save_dir=save_dir_)

    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_v3_train_300K'
    #
    # samples_per_room = 100000
    # generate_n_rirs_v3(samples_per_room, L=[3, 4, 3], desc='small', save_dir=save_dir_)
    # generate_n_rirs_v3(samples_per_room, L=[10, 9, 3], desc='large', save_dir=save_dir_)
    # generate_n_rirs_v3(samples_per_room, L=[2, 10, 2.5], desc='hall', save_dir=save_dir_)

    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_v3_benchmark_3K'
    #
    # samples_per_room = 1000
    # generate_n_rirs_v3(samples_per_room, L=[3, 4, 3], desc='small', save_dir=save_dir_)
    # generate_n_rirs_v3(samples_per_room, L=[10, 9, 3], desc='large', save_dir=save_dir_)
    # generate_n_rirs_v3(samples_per_room, L=[2, 10, 2.5], desc='hall', save_dir=save_dir_)

    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_v3_benchmark_300_ood_rooms'
    #
    # samples_per_room = 100
    # generate_n_rirs_v3(samples_per_room, L=[3, 2, 2], desc='small', save_dir=save_dir_)
    # generate_n_rirs_v3(samples_per_room, L=[12, 12, 4], desc='large', save_dir=save_dir_)
    # generate_n_rirs_v3(samples_per_room, L=[2, 5, 3], desc='hall', save_dir=save_dir_)

    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_v3.1_train_300K'
    #
    # samples_per_room = 100000
    # generate_n_rirs_v3(samples_per_room, L=[3, 4, 3], desc='small', normalize_rms=True, save_dir=save_dir_, version="v3.1")
    # generate_n_rirs_v3(samples_per_room, L=[10, 9, 3], desc='large', normalize_rms=True, save_dir=save_dir_, version="v3.1")
    # generate_n_rirs_v3(samples_per_room, L=[2, 10, 2.5], desc='hall', normalize_rms=True, save_dir=save_dir_, version="v3.1")

    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_v3.1_benchmark_3K'
    # samples_per_room = 1000
    # generate_n_rirs_v3(samples_per_room, L=[3, 4, 3], desc='small', normalize_rms=True, save_dir=save_dir_, version="v3.1", verbose=False)
    # generate_n_rirs_v3(samples_per_room, L=[10, 9, 3], desc='large', normalize_rms=True, save_dir=save_dir_, version="v3.1", verbose=False)
    # generate_n_rirs_v3(samples_per_room, L=[2, 10, 2.5], desc='hall', normalize_rms=True, save_dir=save_dir_, version="v3.1", verbose=False)

    # save_dir_ = f'/{HOME}/asr/scratch/carir/generated_rirs_v3.1_benchmark_300_ood_rooms'
    # samples_per_room = 100
    # generate_n_rirs_v3(samples_per_room, L=[3, 2, 2], desc='small', normalize_rms=True, save_dir=save_dir_, version="v3.1", verbose=False)
    # generate_n_rirs_v3(samples_per_room, L=[12, 12, 4], desc='large', normalize_rms=True, save_dir=save_dir_, version="v3.1", verbose=False)
    # generate_n_rirs_v3(samples_per_room, L=[2, 5, 3], desc='hall', normalize_rms=True, save_dir=save_dir_, version="v3.1", verbose=False)

    # kwargs = dict(receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist = 0.5,
    #               beta = [0.8, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.2_train_300K',
    #               n_samples_to_generate = 100000,
    #               version="v3.2")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin = [(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin = [(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin = [(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)
    #
    # kwargs = dict(receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist = 0.5,
    #               beta = [0.8, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.2_benchmark_3K',
    #               n_samples_to_generate = 1000,
    #               version="v3.2")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin = [(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin = [(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin = [(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)

    # kwargs = dict(receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist = 0.5,
    #               beta = [0.8, 0.95],
    #               normalize_rms=True,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.3_train_300K',
    #               n_samples_to_generate = 100000,
    #               version="v3.3")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin = [(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin = [(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin = [(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)

    # kwargs = dict(receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist = 0.5,
    #               beta = [0.8, 0.95],
    #               normalize_rms=True,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.3_benchmark_3K',
    #               n_samples_to_generate = 1000,
    #               version="v3.3")
    # # ranges:
    # # small = [(1.5, 4), (2.5, 4.5), (2, 3.5)]
    # # large = [(6, 13), (6,12), (2,5)]
    # # hall = [(1,3), (7,13), (2,3.5)]
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin = [(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin = [(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin = [(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)

    # kwargs = dict(receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist = 0.5,
    #               beta = [0.1, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.4_train_300K',
    #               n_samples_to_generate = 100000,
    #               version="v3.4")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin = [(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin = [(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin = [(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)
    #
    # kwargs = dict(receiver_from_wall_margin = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist = 0.5,
    #               beta = [0.1, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.4_benchmark_3K',
    #               n_samples_to_generate = 1000,
    #               version="v3.4")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin = [(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin = [(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin = [(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)

    # kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist=0.5,
    #               beta=[0.5, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.6_train_300K',
    #               n_samples_to_generate=100000,
    #               version="v3.6")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin=[(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin=[(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin=[(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)
    #
    # kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist=0.5,
    #               beta=[0.5, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v3.6_benchmark_3K',
    #               n_samples_to_generate=1000,
    #               version="v3.6")
    # generate_n_rirs_v3(L=[3, 4, 3], wall_margin=[(1.5, 1), (1.5, 0.5), (1, 0.5)], desc='small', **kwargs)
    # generate_n_rirs_v3(L=[10, 9, 3], wall_margin=[(4, 3), (3, 3), (1, 2)], desc='large', **kwargs)
    # generate_n_rirs_v3(L=[2, 10, 2.5], wall_margin=[(1, 1), (3, 3), (0.5, 1)], desc='hall', **kwargs)


    # kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist=0.5,
    #               beta=0.9,
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v4_train_300K',
    #               n_samples_per_room=5000,
    #               version="v4")
    # generate_n_rirs_v4(L=([1.5, 2.5, 2.5], [4, 4.5, 3]), wall_margin=[1, 1, 0.5], desc='small', **kwargs)  # 16 rooms
    # generate_n_rirs_v4(L=([6, 6, 2.5], [13, 12, 4]), wall_margin=[1, 2, 1], desc='large', **kwargs)  # 52 rooms
    # generate_n_rirs_v4(L=([1, 7, 2.5], [3, 13, 3.5]), wall_margin=[1, 1, 1], desc='hall', **kwargs)  # 42 rooms
    #
    # kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist=0.5,
    #               beta=0.9,
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v4_benchmark_3K',
    #               n_samples_per_room=100,
    #               version="v4")
    # generate_n_rirs_v4(L=([1.5, 2.5, 2.5], [4, 4.5, 3]), wall_margin=[1, 1, 0.5], desc='small', **kwargs)  # 16 rooms
    # generate_n_rirs_v4(L=([6, 6, 2.5], [13, 12, 4]), wall_margin=[1, 2, 1], desc='large', **kwargs)  # 52 rooms
    # generate_n_rirs_v4(L=([1, 7, 2.5], [3, 13, 3.5]), wall_margin=[1, 1, 1], desc='hall', **kwargs)  # 42 rooms

    # kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist=0.5,
    #               beta=[0.8, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v4_1_train_300K',
    #               n_samples_per_room=5000,
    #               version="v4.1")
    # generate_n_rirs_v4(L=([1.5, 2.5, 2.5], [4, 4.5, 3]), wall_margin=[1, 1, 0.5], desc='small', **kwargs)  # 16 rooms
    # generate_n_rirs_v4(L=([6, 6, 2.5], [13, 12, 4]), wall_margin=[1, 2, 1], desc='large', **kwargs)  # 52 rooms
    # generate_n_rirs_v4(L=([1, 7, 2.5], [3, 13, 3.5]), wall_margin=[1, 1, 1], desc='hall', **kwargs)  # 42 rooms
    #
    # kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
    #               minimal_r_s_dist=0.5,
    #               beta=[0.8, 0.95],
    #               normalize_rms=False,
    #
    #               save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v4_1_benchmark_3K',
    #               n_samples_per_room=100,
    #               version="v4.1")
    # generate_n_rirs_v4(L=([1.5, 2.5, 2.5], [4, 4.5, 3]), wall_margin=[1, 1, 0.5], desc='small', **kwargs)  # 16 rooms
    # generate_n_rirs_v4(L=([6, 6, 2.5], [13, 12, 4]), wall_margin=[1, 2, 1], desc='large', **kwargs)  # 52 rooms
    # generate_n_rirs_v4(L=([1, 7, 2.5], [3, 13, 3.5]), wall_margin=[1, 1, 1], desc='hall', **kwargs)  # 42 rooms

    kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                  source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                  minimal_r_s_dist=0.5,
                  beta=0.9,
                  normalize_rms=False,
                  save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v5_train_300K',
                  n_samples_per_room=5000,
                  version="v5")
    generate_n_rirs_v5(L=([1.5, 2.5, 2.5], [4, 4.5, 3]), wall_margin=[1, 1, 0.5], desc='small', **kwargs)  # 16 rooms
    generate_n_rirs_v5(L=([6, 6, 2.5], [13, 12, 4]), wall_margin=[1, 2, 1], desc='large', **kwargs)  # 52 rooms
    generate_n_rirs_v5(L=([1, 7, 2.5], [3, 13, 3.5]), wall_margin=[1, 1, 1], desc='hall', **kwargs)  # 42 rooms

    kwargs = dict(receiver_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                  source_from_wall_margin=[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)],
                  minimal_r_s_dist=0.5,
                  beta=0.9,
                  normalize_rms=False,
                  save_dir=f'/{HOME}/asr/scratch/carir/generated_rirs/rirs_v5_benchmark_3K',
                  n_samples_per_room=100,
                  version="v5")
    generate_n_rirs_v5(L=([1.5, 2.5, 2.5], [4, 4.5, 3]), wall_margin=[1, 1, 0.5], desc='small', **kwargs)  # 16 rooms
    generate_n_rirs_v5(L=([6, 6, 2.5], [13, 12, 4]), wall_margin=[1, 2, 1], desc='large', **kwargs)  # 52 rooms
    generate_n_rirs_v5(L=([1, 7, 2.5], [3, 13, 3.5]), wall_margin=[1, 1, 1], desc='hall', **kwargs)  # 42 rooms

    pass
