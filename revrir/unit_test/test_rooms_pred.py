import os
import torch
import pickle as pkl
import lsb_release

from ..eval.rooms_pred import roomsPred
from ..utils import get_audio, get_audio_paths, HOME

unit_test_prefix = f"/{HOME}/asr/scratch/carir/export_dir/models"
gitlab_hash_versions = {'Ubuntu 20.04.5 LTS': 'jab5fdfd6aac3cebae60f18a53760d402d0d402d',  # gitlabci runners
                        'Ubuntu 22.04.2 LTS': 'jab5fdfd6aac3cebae60f18a53760d402d0d402d',  # gitlabci-srv
                        'Ubuntu 20.04.4 2022.04.07 LTS (Cubic 2022-04-07 16:55)': '8ab5fdfd6aac3cebae60f18a53760d402dacdd1f'}  # blade30a
unit_test_gitlab_prefix = f"/orcam/devops/data/assets/dev/orcam/devops/data/assets/carir-{gitlab_hash_versions[lsb_release.get_distro_information()['DESCRIPTION']]}"


def test_3_rooms_model():
    dir_path = "open_slr_68k_ckpt_49_2"
    model_path = f"{unit_test_prefix}/{dir_path}/stripped_model_49.pkl"
    expected_res_path = os.path.join(os.path.dirname(__file__), "expected_result_3_rooms_model.pkl")
    if not os.path.isfile(model_path):
        model_path = f"{unit_test_gitlab_prefix}/{dir_path}/stripped_model_49.pkl"
        expected_res_path = f"{unit_test_gitlab_prefix}/{dir_path}/{os.path.basename(expected_res_path)}"
    running_function(model_path, expected_res_path)


def test_train_300K_v4_bins_ckpt_350():
    dir_path = "train_300K_v4_bins_ckpt_350"
    model_path = f"{unit_test_prefix}/{dir_path}/stripped_model_50.pkl"
    expected_res_path = os.path.join(os.path.dirname(__file__), "expected_result_v4_bins_model.pkl")
    if not os.path.isfile(model_path):
        model_path = f"{unit_test_gitlab_prefix}/{dir_path}/stripped_model_50.pkl"
        expected_res_path = f"{unit_test_gitlab_prefix}/{dir_path}/{os.path.basename(expected_res_path)}"
    running_function(model_path, expected_res_path)


def test_ast_no_t_ckpt_350_model():
    dir_path = "ast_no_t_ckpt_350"
    model_path = f"{unit_test_prefix}/{dir_path}/stripped_model_50.pkl"
    expected_res_path = os.path.join(os.path.dirname(__file__), "expected_result_ast_no_t_model.pkl")
    if not os.path.isfile(model_path):
        model_path = f"{unit_test_gitlab_prefix}/{dir_path}/stripped_model_50.pkl"
        expected_res_path = f"{unit_test_gitlab_prefix}/{dir_path}/{os.path.basename(expected_res_path)}"
    running_function(model_path, expected_res_path)


def running_function(model_path, expected_res_path):
    model = roomsPred(model_path, os.path.dirname(model_path))
    audio_paths = get_audio_paths("test")

    expected_res = pkl.load(open(expected_res_path, "rb"))
    # res = {}
    for a_p in audio_paths:
        audio = get_audio(a_p, model.get_fs())
        emb = model.get_audio_emb_from_audio(audio)
        classes = model.classify_from_audio_in_chunks(audio)

        # res[f'{os.path.basename(a_p)}_emb'] = emb
        # res[f'{os.path.basename(a_p)}_classes'] = classes
        assert torch.allclose(emb, expected_res[f'{os.path.basename(a_p)}_emb'], atol=4e-08)
        assert torch.allclose(classes, expected_res[f'{os.path.basename(a_p)}_classes'])

    # pkl.dump(res, open(expected_res_path, "wb"))


if __name__ == '__main__':
    test_train_300K_v4_bins_ckpt_350()
    test_ast_no_t_ckpt_350_model()
    test_3_rooms_model()
