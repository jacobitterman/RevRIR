import json
import os
import math
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

import transformers
from transformers import Trainer, TrainingArguments, set_seed

from ..data.dataset_v2 import CARIR_dataset as CARIR_dataset_v2
from ..data.dataset_v2 import collate_fn as collate_fn_v2
from ..data.dataset_v3 import CARIR_dataset_V3
from ..data.dataset_v3 import collate_fn_v3
from ..data.dummy_dataset import DummyDataset
from ..data.data_utils import get_sim_mat


from ..models.modeling_carir import CarirModel, CarirPreTrainedModel, CarirRtfModel
from ..models.instruct_carir import CarirV2, PretrainedAST
from ..processing_carir import CarirProcessor
from ..configuration_carir import CarirConfig, CarirRtfConfig

from ..utils import get_audio_paths, makedirs_p


def classification_score(audio_emb, rtf_emb, gt_idxs, sim_mat=None):
    sim = torch.nn.functional.cosine_similarity(audio_emb[None, :, :],
                                                rtf_emb[:, None, :], dim=-1).T
    diag_mean = sim.diag().mean()
    accuracy = accuracy_score(sim.argmax(-1).detach().cpu().numpy(), gt_idxs)
    rank_score = gt_idxs.shape[0] - ((sim.argsort(-1).detach().cpu() == torch.tensor(gt_idxs[:, None])).nonzero()[:, 1] / 1.0).mean()

    submax_non_room_mean, sim_of_same_room = None, None
    if sim_mat is not None:
        sim_of_same_room = sim[sim_mat > 0][torch.triu_indices(sim.shape[0], sim.shape[1], 1)].mean().item()
        sim[sim_mat > 0] = 0
        submax_non_room_mean = sim.max(dim=1).values.mean().item()

    return accuracy, rank_score.item(), diag_mean.item(), sim_of_same_room, submax_non_room_mean


class trainerCallback(transformers.TrainerCallback):
    def __init__(self, training_dir, output_keymap = {}):
        from tensorboardX import SummaryWriter
        self.logger = SummaryWriter(os.path.join(training_dir, "eval_logs"))
        self.logging_prefix = "eval/"
        self.output_keymap = output_keymap  # for backword compatiblity

    @staticmethod
    def md_to_str(x, prefix):
        L = x["L_new"] if "L_new" in x else x["L"]
        s = f'{prefix}_r_{x["r"]}_s_{x["s"]}_L_{L}_t60_{x["reverberation_time"]}'
        return s

    def on_evaluate(self, args, state, control, **kwargs):
        kwargs["model"].eval()
        step = state.global_step
        sum_acc = 0
        sum_rank = 0
        embeddings = []
        embeddings_md = []
        num_batches = len(kwargs["eval_dataloader"])
        all_rooms = []
        batch_size = -1
        with torch.no_grad():
            for batch in tqdm.tqdm(kwargs["eval_dataloader"], total=num_batches, desc=f"eval in step {step}"):
                batch['return_loss'] = False
                output = kwargs["model"](**batch)
                for k, v in self.output_keymap.items():  # for backword compatiblity
                    if v in output:
                        output[k] = output[v]
                labels = np.arange(output["audio_embeds"].shape[0])  #TODO: support labels that come from the batch?
                acc, rank, _, _, _ = classification_score(output["audio_embeds"], output["rtf_embeds"], labels)
                sum_acc = sum_acc + acc
                sum_rank = sum_rank + rank
                rir_md = [self.md_to_str(x, y) for x, y in zip(batch["rir_md"], batch["audio_rir_corr"])]

                embeddings.append(torch.cat((output["audio_embeds"], output["rtf_embeds"])).detach().cpu().numpy())
                embeddings_md = embeddings_md + batch["audio_rir_corr"] + rir_md
                all_rooms.append(batch['v4_room_indx'])

        all_embeddings = np.vstack(embeddings)
        labels = np.arange(all_embeddings.shape[0] // 2)
        batch_size = embeddings[0].shape[0] // 2
        all_rooms = np.hstack(all_rooms)
        sim_mat = get_sim_mat(all_rooms)
        sim_mat = (sim_mat * all_rooms)
        from orcam_utils.debug_utils.embed import embed; embed(locals(), globals())
        total_acc, total_rank, diag_mean, sim_of_same_room, submax_non_room_mean = classification_score(torch.Tensor(np.vstack([a[:batch_size] for a in embeddings])),
                                                     torch.Tensor(np.vstack([a[batch_size:] for a in embeddings])),
                                                     labels,
                                                     sim_mat,
                                                     )
        self.logger.add_embedding(np.vstack(embeddings),
                                  metadata = embeddings_md,
                                  tag = self.logging_prefix + "embedding_space",
                                  global_step=step)
        mean_accuracy = sum_acc / num_batches
        mean_rank = sum_rank / num_batches
        self.logger.add_scalar(self.logging_prefix + "accuracy", mean_accuracy, global_step=step)
        self.logger.add_scalar(self.logging_prefix + "rank_score", mean_rank, global_step=step)
        self.logger.add_scalar(self.logging_prefix + "total_accuracy", total_acc, global_step=step)
        self.logger.add_scalar(self.logging_prefix + "total_rank_score", total_rank, global_step=step)
        self.logger.add_scalar(self.logging_prefix + "sim_diag_mean", diag_mean, global_step=step)
        self.logger.add_scalar(self.logging_prefix + "sim_same_room_mean", sim_of_same_room, global_step=step)
        self.logger.add_scalar(self.logging_prefix + "submax_non_room_mean", submax_non_room_mean, global_step=step)
        print(f"{mean_accuracy=:.3f}, {mean_rank=:.3f}, {total_acc=:.3f}, {total_rank=:.3f}, {diag_mean=:.3f}, {sim_of_same_room=:.3f}, "
              f"{submax_non_room_mean=:.3f}")

    def on_save(self, args, state, control, **kwargs):
        pass

    def __del__(self):
        self.logger.close()


def train(training_dataset = 'libri-train-100',
          training_rirs = "v3.1_train_300K",
          eval_dataset = 'libri-dev-clean',
          eval_rirs = "v3.1_benchmark_3K",
          n_samples_training = 20000,
          n_samples_eval = 2000,
          batch_size = 50,
          output_dir=None,
          dataloader_num_workers = 20,
          num_train_epochs=500,
          dummy_samples = -1,
          learning_rate=1e-5,
          pretrained_dir = "/tmp/",
          compress_audio_prob = 0,
          logits_scale_a=-0.0133,
          logits_scale_r=2.9264,
          eval_steps=2000,
          skip_rir_trad_feat = False,
          resume_from_checkpoint=False,
          ):
    makedirs_p(output_dir)
    try:
        from ..utils import code_logger_wrapper
        loggs_path = code_logger_wrapper(path=output_dir, run_name='logger')  # save code + dependencies + log sys.stdout
    except ImportError:
        pass

    # moved from ~/.cache/huggingface/hub/models--carir/snapshots/8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a
    # the weights are of CLAP model (laion--clap-htsat-unfused)
    from ..utils import get_generater_rirs_paths
    rir_paths, md_paths = get_generater_rirs_paths(training_rirs)
    eval_rir_paths, eval_md_paths = get_generater_rirs_paths(eval_rirs)

    # load processor and model
    pretrained_dir = os.path.expanduser(pretrained_dir)
    carir_processor = CarirProcessor.from_pretrained(pretrained_dir)
    name_conversion_map, output_keymap = {}, {}
    if "MIT" in pretrained_dir and "ast" in pretrained_dir:
        embedder1 = PretrainedAST(carir_processor, pretrained_dir, ignore_mismatched_sizes=True)
        rtf_config = CarirRtfConfig.from_json_path(os.path.join(pretrained_dir, "config.json"))
        embedder2 = CarirRtfModel(rtf_config)
        carir_model = CarirV2(embedder1,
                              embedder2,
                              logits_scale_a,
                              logits_scale_r,
                              dim_latent = rtf_config.projection_dim,
                              )
        name_conversion_map = {'input_audio_features': 'input_em_1', 'input_rtf_features': 'input_em_2'}
        output_keymap = {'audio_embeds': 'embedder1_latents_normed', 'rtf_embeds': 'embedder2_latents_normed'}
    else:
        carir_model = CarirModel.from_pretrained(pretrained_dir)
        rtf_config = carir_model.rtf_config
    data_type = "v2"
    if hasattr(carir_model, "orig_loss") and not carir_model.orig_loss:
        data_type = "v3"
    assert data_type in ["v2", "v3"], data_type

    # dataset
    train_audio_paths = get_audio_paths(training_dataset)[:n_samples_training]
    print(f'selected {n_samples_training} audio samples for training, in practice we got {len(train_audio_paths)}')
    eval_audio_paths = get_audio_paths(eval_dataset)[:n_samples_eval]
    print(f'selected {n_samples_eval} audio samples for eval, in practice we got {len(eval_audio_paths)}')

    fs = carir_processor.feature_extractor.sampling_rate

    dataset_class = None
    collate_fn = None
    if data_type == "v2":
        dataset_class = CARIR_dataset_v2
        collate_fn = collate_fn_v2
        print("using dataset v2")
    elif data_type == "v3":
        dataset_class = CARIR_dataset_V3
        collate_fn = collate_fn_v3
        print("using dataset v3")
    else:
        assert 0

    dataset = dataset_class(audio_paths=train_audio_paths,
                            rir_paths=rir_paths,
                            md_paths=md_paths,
                            fs=fs,                     # Hz
                            mode = "train",
                            compress_audio_prob = compress_audio_prob,
                            # rms_augmentation_prob=-1,
                            # rms_factor_augmentation_range=(2, 2.5),
                            sample_rooms_by_bins = False,
                            skip_rir_trad_feat=skip_rir_trad_feat,
                            )
    eval_dataset = CARIR_dataset_v2(audio_paths=eval_audio_paths,  # should be v2 in order for ranking and accuracy calc
                                    rir_paths=eval_rir_paths,
                                    md_paths=eval_md_paths,
                                    fs=fs,                     # Hz
                                    mode = "eval",
                                    compress_audio_prob=compress_audio_prob,
                                    sample_rooms_by_bins=False,
                                    skip_rir_trad_feat=skip_rir_trad_feat,
                                    )
    if dummy_samples > 0:
        dataset = DummyDataset(dataset, dummy_samples)

    # Hyperparameters
    print(f"batch_size is: {batch_size}")

    # Create a data loader to batch and shuffle the data
    print(f'max_n_rir: {rtf_config.in_dim}')
    if carir_processor.rtf_extractor.process_time_domain:
        nsample_rir = rtf_config.in_dim
    else:
        nsample_rir = (rtf_config.in_dim - 1) * 2
    dataset.worker_init_function(0)  # init it to get dataset.max_rir_length
    assert nsample_rir >= dataset.max_rir_length, dataset.max_rir_length
    collate_fn_p = collate_fn(processor=carir_processor,
                              fs=fs,
                              nsample_rir = nsample_rir,
                              name_conversion_map=name_conversion_map,
                              )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.05,
        logging_steps=10,
        push_to_hub=False,
        remove_unused_columns=False,
        dataloader_num_workers = dataloader_num_workers,
        eval_steps = eval_steps,
        save_steps=2000,
        dataloader_drop_last=True,
        # lr_scheduler_type = "constant"
    )

    tc = trainerCallback(output_dir, output_keymap)

    trainer = Trainer(
        carir_model,
        args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator = collate_fn_p,
        callbacks=[tc],
        worker_init_fn=dataset.worker_init_function
    )
    # trainer.accelerator.init_trackers(os.path.join(output_dir, "logs"))

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


import fire
if __name__ == '__main__':
    fire.Fire()
