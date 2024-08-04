import os
import json
import shutil
import random
import traceback
import pickle as pkl
import yt_dlp as youtube_dl
from tqdm import tqdm
from typing import Any, List, Union, Optional, Dict, Tuple

from ..utils import makedirs_p


def get_videos_in_playlist(url: str,
                           retry: bool=True
                           ) -> Optional[Dict]:
    """
    return all videos (entries) in a playlist or channel
    :param url: url of playlist or channel
    :param retry: retry if youtube dl return object with url instead of entries
    :return: dict
    """
    ydl_opts = dict(extract_flat=True)
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        r = ydl.extract_info(url, download=False)
        if 'entries' in r:
            return r
        elif retry:
            return get_videos_in_playlist(r['url'], False)
        else:
            return None


def extract_video_ids_from_channels(channels: List[str]) -> Tuple[List, Dict]:
    """
    extract all video id to download from a list of mixed video and channels
    :param channels: list of mixed channels / playlists / video id
    :return: tuple(list of video id, dict of video id per channel)
    """
    all_vids = dict()
    all_vids_flat = []
    for id in channels:
        if len(id) == 11:  # video id
            all_vids_flat.append(id)
            continue
        if id.startswith('U'):  # channel
            url = 'https://www.youtube.com/channel/' + id
        else:  # playlist
            url = 'https://www.youtube.com/playlist?list=' + id

        # extract all video in the pl / channel
        videos = get_videos_in_playlist(url, retry=True)
        if videos is not None and 'entries' in videos:
            for e in videos['entries']:
                if id in all_vids:
                    all_vids[id].add(e['id'])
                else:
                    all_vids[id] = {e['id']}

    # extract all video if from the dict
    all_vids_flat = all_vids_flat + [vid_id for k, vids in all_vids.items() for vid_id in vids]
    return all_vids_flat, all_vids


def save_extract_info_to_json(extract_info: Optional[Dict],
                              save_dir: str,
                              video_id: str
                              ) -> Optional[Dict[str, Any]]:
    """
    extract values of fields from list EXTRACT_FIELDS in the extract_info youtube result.
    if field does not exist in the dict - puts None in the value
    :param extract_info: youtube download's result
    :param save_dir: save path
    :param video_id: video id
    :return: dict of the fields and their's value
    """
    if extract_info is not None:
        json_filename = save_dir + '/' + video_id + '.json'
        with open(json_filename, 'wt') as f:
            json.dump(extract_info, f)
        return extract_info
    else:
        return None


def download_video_ids(output_path: str,
                       all_vids_list: List[str],
                       language_codes: Union[List[str], str],
                       download_audio: bool=True,
                       write_description=True,
                       ) -> None:
    def move_desc_file(move_file: bool, tmp_down_dir: str, v_id: str, s_dir: str):
        # move description file
        if move_file:
            tmp_desc_filename = tmp_down_dir + '/' + v_id + '.description'
            desc_filename = (s_dir + '/' + v_id + '.description')
            if os.path.isfile(tmp_desc_filename):
                shutil.move(tmp_desc_filename, desc_filename)

    tmp_download_dir = output_path + '/tmp'
    print(f"download dir: {tmp_download_dir}")
    print(f"there are total {len(all_vids_list)} vids")

    results_save_path_output = output_path + "/results_dicts.pkl"

    curr_results = []

    # shuffle video for not going to youtube sequentially
    random.shuffle(all_vids_list)

    outtmpl = tmp_download_dir + '/%(id)s.%(ext)s'
    # dict for youtube-dl options
    ydl_opts = dict(subtitleslangs=language_codes,
                    format='bestaudio',
                    outtmpl=outtmpl,
                    noplaylist=True,
                    skip_download=False,
                    writedescription=write_description)

    print(f"youtube-dl options: {ydl_opts}")
    # go over video ids
    for indx, video_id in enumerate(tqdm(all_vids_list, desc="downloading")):
        extract_info = None
        try:
            print(f'Extracting info {video_id}')
            # make youtube object
            url = f'https://www.youtube.com/watch?v={video_id}'
            # download
            ydl_extract_info = youtube_dl.YoutubeDL(dict(ydl_opts,
                                                     writesubtitles=True,
                                                     writeautomaticsub=True))
            extract_info = ydl_extract_info.extract_info(url, download=True)

            # find which one of language code downloaded
            downloaded_vtt_codes = []
            tmp_vtt_filename = None
            for code in language_codes[::-1]:
                tmp_vtt_filename: str = tmp_download_dir + '/' + video_id + '.' + code + '.vtt'
                if os.path.exists(tmp_vtt_filename):
                    downloaded_vtt_codes.append({'code': code,
                                                 'tmp_vtt_filename': tmp_vtt_filename})

            curr_chanel = extract_info['channel_id'] if extract_info is not None else "unknown"
            # make save dir by channel
            save_dir = os.path.join(output_path, curr_chanel)
            if not os.path.isdir(save_dir):
                makedirs_p(save_dir, mode=0o777)

            # save json of fields
            save_extract_info_to_json(extract_info, save_dir, video_id)

            # move audio file
            file_size = False
            audio_filename = None
            if extract_info is None or extract_info['ext'] is None:
                ext = ''
            else:
                ext = extract_info['ext'] #type: ignore
            tmp_audio_filename = tmp_download_dir + '/' + video_id + '.' + ext
            audio_filename = (save_dir + '/' + video_id + '.' + ext)
            file_size = os.path.getsize(tmp_audio_filename)
            shutil.move(tmp_audio_filename, audio_filename)

            vtt_filename = None
            # move vtt file from tmp
            if downloaded_vtt_codes:
                vtt_filename = save_dir + '/' + video_id + '.' + downloaded_vtt_codes[0]['code'] + '.vtt'
                shutil.move(tmp_vtt_filename, vtt_filename)

            move_desc_file(write_description, tmp_download_dir, video_id, save_dir)

            # add new dict for the video
            new_doc: Dict[str, Union[str, bool]] = {# type: ignore
                'audio_path': audio_filename,
                'vtt_basename': vtt_filename,
                'duration': extract_info['duration'] if extract_info is not None else None,
                'subtitles_lang': downloaded_vtt_codes[0]['code'] if downloaded_vtt_codes else None,
                'uploader_id': extract_info['uploader_id'] if extract_info is not None else None,
                'channel_id': extract_info['channel_id'] if extract_info is not None else None,
                'uploader': extract_info['uploader'] if extract_info is not None else None,
                'filesize': file_size,
            }
            curr_results.append((video_id, new_doc))
            continue
        # handle errors
        except youtube_dl.utils.DownloadError as ydl_ex:
            if hasattr(ydl_ex, 'args') and 'HTTP Error 429' in ydl_ex.args[0]:
                LOG.critical("you have been blocked from downloading")
                assert 0
            message = 'no message'
            if hasattr(ydl_ex, 'args'):
                message = ydl_ex.args[0]
            new_doc = {'ERROR': 'download error', 'message': message}
            curr_results.append((video_id, new_doc))
            continue
        except IOError as ioerr:
            message = 'no message'
            if hasattr(ioerr, 'args'):
                message = ioerr.args[0]
            new_doc = {'ERROR': 'io error', 'message': message}
            curr_results.append((video_id, new_doc))
            continue
        except Exception as e:
            print(f"error occurred: {e}. traceback for this exception:")
            print(traceback.format_exc())
            message = 'no message'
            print("exception")
            traceback.print_exc()
            if hasattr(e, 'args'):
                message = e.args[0]
            new_doc = {'ERROR': 'unknown error', 'message': message}
            curr_results.append((video_id, new_doc))
            continue

    pkl.dump(curr_results, open(results_save_path_output, 'wb'))


def main():
    # vids = ["UCVapw_3nx05RRiq1cbDINPg",  # exctract channel
    #         "m9wYyBElxoQ"]
    # vids, info = extract_video_ids_from_channels(vids)
    from ..utils import HOME
    out_path = f"/{HOME}/asr/scratch/carir/audio_data/audiobooks/raw_data"
    # pkl.dump(vids, open(os.path.join(out_path, "channels.pkl"), 'wb'))
    vids = pkl.load(open(os.path.join(out_path, "channels.pkl"), 'rb'))
    download_video_ids(out_path, vids, ["ar"])


def split_to_segments(seg_size_sec = 10, fs = 8000):
    from ..utils import HOME, get_audio
    from scipy.io import wavfile

    raw_path = f"/{HOME}/asr/scratch/carir/audio_data/audiobooks/raw_data"
    segments_path = os.path.join(os.path.dirname(raw_path), "segments")
    channels = [d for d in os.listdir(raw_path) if d.startswith('UCV')]
    for channel in channels:
        c_dir = os.path.join(raw_path, channel)
        new_c_dir = os.path.join(segments_path, channel)
        makedirs_p(new_c_dir)
        audios = [d for d in os.listdir(c_dir) if d.endswith(('webm', "m4a"))]
        for aud in tqdm(audios, total=len(audios)):
            aud_path = os.path.join(c_dir, aud)
            new_aud_path = os.path.join(new_c_dir, aud.split(".")[0])
            makedirs_p(new_aud_path)
            aud = get_audio(aud_path, fs)
            print("loaded")
            audio_len = len(aud)
            num_chunks = (audio_len / fs) / seg_size_sec

            # split to chunks and save:
            i = 0
            while i < num_chunks:
                chunk = aud[i*seg_size_sec*fs : min((i+1)*seg_size_sec*fs, audio_len)]
                wavfile.write(os.path.join(new_aud_path, f"seg_{i}.wav"), fs, chunk)
                i = i + 1


def build_binary_from_segments(fs = 8000):
    from ..utils import HOME, get_audio, IndexedBinary, encode_buff
    from scipy.io import wavfile

    raw_path = f"/{HOME}/asr/scratch/carir/audio_data/audiobooks/raw_data"
    segments_path = os.path.join(os.path.dirname(raw_path), "segments")
    binaries_path = os.path.join(os.path.dirname(raw_path), "binaries")
    makedirs_p(binaries_path)
    channels = [d for d in os.listdir(segments_path) if d.startswith('UCF')]
    for channel in channels:
        c_dir = os.path.join(segments_path, channel)
        bin_path = os.path.join(binaries_path, f"{channel}.bin")
        ids = os.listdir(c_dir)
        with IndexedBinary(bin_path, "w") as bin:
            for i, video_id in enumerate(ids):
                id_dir = os.path.join(c_dir, video_id)
                audios = [d for d in os.listdir(id_dir) if d.endswith("wav")]
                for aud_bn in tqdm(audios, total=len(audios), desc=f"video_id {video_id} {i}/{len(ids)}"):
                    aud_path = os.path.join(id_dir, aud_bn)
                    aud = get_audio(aud_path, fs)
                    from orcam_utils.debug_utils.embed import embed; embed(locals(), globals())
                    segment_idx = aud_bn.split("_")[1].split(".")[0]
                    buff = {
                        'audio': aud,
                        'fs': fs,
                        'duration': len(aud) / fs,
                        'v_id': video_id,
                        'idx': segment_idx,
                        'audio_path': aud_path,
                    }
                    bin.write(encode_buff(buff))




if __name__ == '__main__':
    build_binary_from_segments()
    # split_to_segments()
    # main()
