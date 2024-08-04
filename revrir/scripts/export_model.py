import os
import json
import torch
import shutil
from ..models.classifier import RtfClassifierConfig
from ..utils import makedirs_p


def export(classifier_model_path, target_dir, strip_unused_params = True):
    makedirs_p(target_dir)
    config_path = os.path.join(os.path.dirname(classifier_model_path), "config.conf")
    config = RtfClassifierConfig.from_dict(json.load(open(config_path, "r")))

    if strip_unused_params:
        from ..eval.rooms_pred import roomsPred
        model = roomsPred(classifier_model_path)
        # remove unused parameters:
        state_dict = torch.load(classifier_model_path, map_location=next(model.model.classify.parameters()).device)
        dict_keys = list(state_dict.keys())
        for k in dict_keys:
            if k.startswith("rtf2hidden_model") or k.startswith("rtf_projection"):
                del state_dict[k]
            else:
                new_name = k.replace('classify.', '')
                if new_name != k:
                    state_dict[new_name] = state_dict[k]
                    del state_dict[k]
        new_model_path = classifier_model_path.replace("model_", "stripped_model_")
        torch.save(state_dict, new_model_path)

    pretrain_path = config.encoder_from_pretrain
    pretrain_model_path = os.path.join(pretrain_path, "pytorch_model.bin")
    pretrain_config_path = os.path.join(pretrain_path, "config.json")
    pretrain_pp_config_path = os.path.join(pretrain_path, "preprocessor_config.json")
    pretrain_rooms_pkl_path = os.path.join(os.path.dirname(classifier_model_path), "rooms.pkl")
    # pretrain_model_path = os.path.join(pretrain_path, "trainer_state.json")
    # pretrain_model_path = os.path.join(pretrain_path, "training_args.bin")

    files_to_export = [config_path,

                       pretrain_model_path,
                       pretrain_config_path,
                       pretrain_pp_config_path,
                       pretrain_rooms_pkl_path,
                       ]

    if strip_unused_params:
        files_to_export.append(new_model_path)
    else:
        files_to_export.append(classifier_model_path)

    for p in files_to_export:
        dst = os.path.join(target_dir, os.path.basename(p))
        print(f"exporting file {p} to {dst}")
        shutil.copy(p, dst)


if __name__ == '__main__':
    from ..utils import HOME
    # export_dir = f"/{HOME}/asr/scratch/carir/export_dir/models/open_slr_68k_ckpt_49_2"
    # model_path = f"/{HOME}/asr/scratch/carir/classifier_training/rooms/train_300K_v3_2_ckpt_68K_from_openslr/model_49.pkl"  # best - 87%

    # export_dir = f"/{HOME}/asr/scratch/carir/export_dir/models/train_300K_v4_bins_ckpt_350"
    # model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_bins_ckpt_350/model_50.pkl"  # 83% on rir, 37% on audio

    export_dir = f"/{HOME}/asr/scratch/carir/export_dir/models/ast_no_t_ckpt_350"
    model_path = f"/{HOME}/asr/scratch/carir/classifier_training/110_rooms/train_300K_v4_ast_no_4_ckpt_350/model_50.pkl"  # 78% on rir, 34% on audio
    export(model_path, export_dir)

    from ..eval.rooms_pred import roomsPred
    r = roomsPred(os.path.join(export_dir, f"stripped_{os.path.basename(model_path)}"))
