import os
import json

import numpy as np
import torch
import pickle as pkl
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime


from ..data.dummy_dataset import DummyDataset
from ..data.classify_datasets import roomClassifierDataset, roomClassifierMCDataset, collate_fn

from ..utils import get_generater_rirs_paths, makedirs_p, get_audio_paths
from ..feature_extraction_carir import RtfFeatureExtractor, CarirFeatureExtractor
from ..models.classifier import EmbClassifier, RtfClassifierConfig


def calc_multi_class_loss(loss_fn, model_output, gt, num_classes, regression=True):
    scores_x, scores_y, scores_z = model_output
    #  PIT loss
    loss_xx = loss_fn(scores_x.squeeze(), gt[:, 0])
    loss_yy = loss_fn(scores_y.squeeze(), gt[:, 1])
    loss_xy = loss_fn(scores_x.squeeze(), gt[:, 1])
    loss_yx = loss_fn(scores_y.squeeze(), gt[:, 0])
    loss_zz = loss_fn(scores_z.squeeze(), gt[:, 2])
    loss1 = loss_xx + loss_yy + loss_zz
    loss2 = loss_xy + loss_yx + loss_zz
    loss = torch.min(loss1, loss2)  # get the minimal loss cause we let the model to swap between x, y
    loss = loss.mean().float()

    argmin = torch.argmin(torch.stack([loss1, loss2]), dim=0)

    batch_size = argmin.shape[0]

    if regression:
        delta_x = torch.stack([loss_xx, loss_xy])[argmin, torch.arange(0, batch_size, dtype=int)]
        delta_y = torch.stack([loss_yy, loss_yx])[argmin, torch.arange(0, batch_size, dtype=int)]
        delta_z = loss_zz
        return loss, delta_x, delta_y, delta_z, argmin
    else:
        num_classes_x, num_classes_y, num_classes_z = num_classes
        real_scores_x = torch.stack([scores_x, scores_y])[
            argmin, torch.arange(0, batch_size, dtype=int)]  # take scores_x using argmin
        real_scores_y = torch.stack([scores_y, scores_x])[
            argmin, torch.arange(0, batch_size, dtype=int)]  # take scores_y using argmin
        acc_x, acc_by_class_x, num_samples_x = calc_acc(real_scores_x.argmax(-1), gt[:, 0], num_classes_x)
        acc_y, acc_by_class_y, num_samples_y = calc_acc(real_scores_y.argmax(-1), gt[:, 1], num_classes_y)
        acc_z, acc_by_class_z, num_samples_z = calc_acc(scores_z.argmax(-1), gt[:, 2], num_classes_z)
        return loss, acc_x, acc_y, acc_z, argmin


def calc_acc(scores, target, num_class, normalize = True):
    acores_argmax = scores.argmax(-1)
    acc = torch.sum(acores_argmax == target)
    if normalize:
        acc = acc / acores_argmax.shape[0]

    rank_score = num_class - ((scores.argsort(-1).detach() == target[:, None]).nonzero()[:, 1] / 1.0).mean()
    acc_by_class = {}
    num_samples_from_each_class = {}
    for i in range(num_class):
        num_samples_from_each_class[i] = torch.sum(target == i)
        acc_by_class[i] = 0
        acc_by_class[i] = torch.sum((acores_argmax == target)[target == i])
        if normalize and num_samples_from_each_class[i] > 0:
            acc_by_class[i] = acc_by_class[i] / num_samples_from_each_class[i]

    return rank_score, acc, acc_by_class, num_samples_from_each_class


def train(num_workers=1,
          epochs=5,
          training_dir = '',
          clasify_target = "room",
          encoder_from_pretrain = None,
          dummy_batches = -1,
          batch_size = 50,
          lr = -1,
          training_rirs="v3.1_train_300K",
          eval_rirs="v3.1_benchmark_3K",
          process_time_domain=False,
          acc_every = 1,
          skip_audio=True,
          train_dataset="",
          eval_dataset = 'libri-dev-clean',
          head_size=None,
          freeze_encoder=True,
          ):
    makedirs_p(training_dir)
    try:
        from ..utils import code_logger_wrapper
        loggs_path = code_logger_wrapper(path=training_dir, run_name='logger')  # save code + dependencies + log sys.stdout
    except ImportError:
        pass

    audio_paths_train, audio_paths_eval = None, None
    multi_class = False
    multi_class_regression = False
    if clasify_target == "room":
        dataset_obj = roomClassifierDataset
        dataset_kwargs = {'skip_audio': skip_audio}
        num_classes = 3
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
    elif clasify_target == "room_v4":
        dataset_obj = roomClassifierDataset
        if not skip_audio:
            audio_paths_train = get_audio_paths(train_dataset)
            audio_paths_eval = get_audio_paths(eval_dataset)
        dataset_kwargs = {'is_v4': True, 'skip_audio': skip_audio, 'audio_paths': audio_paths_train, 'sample_rooms_by_bins': False}
        num_classes = 110
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
    elif clasify_target == "room_dim":
        w_stride = [0.25, 0.25, 0.25]
        w_min_length = [1, 1, 2]
        num_classes_x = int((13 - w_min_length[0]) / w_stride[0])
        num_classes_y = int((13 - w_min_length[1]) / w_stride[1])
        num_classes_z = int((5 - w_min_length[2]) / w_stride[2])
        num_classes = [num_classes_x, num_classes_y, num_classes_z]
        dataset_obj = roomClassifierMCDataset
        dataset_kwargs = {'return_room_dim_as_is': False, 'w_min_length': w_min_length, 'w_stride': w_stride}
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        multi_class = True
    elif clasify_target == "room_dim_regression":
        num_classes = [1, 1, 1]
        loss_fn = nn.MSELoss(reduction='none')
        dataset_obj = roomClassifierMCDataset
        dataset_kwargs = {'return_room_dim_as_is': True}
        multi_class, multi_class_regression = True, True
    else:
        assert 0

    rir_paths, md_paths = get_generater_rirs_paths(training_rirs)
    dataset = dataset_obj(rir_paths=rir_paths,
                          md_paths=md_paths,
                          **dataset_kwargs)
    dataset_kwargs['audio_paths'] = audio_paths_eval
    eval_rir_paths, eval_md_paths = get_generater_rirs_paths(eval_rirs)
    eval_dataset = dataset_obj(rir_paths=eval_rir_paths,
                               md_paths=eval_md_paths,
                               mode="eval",
                               **dataset_kwargs)

    total_classes = len(num_classes) if isinstance(num_classes, list) else num_classes
    assert dataset.num_classes == total_classes,  total_classes

    config = RtfClassifierConfig(5201, 768, 768, 512,
                                 "relu", num_classes, -1, freeze_encoder, encoder_from_pretrain,
                                 dummy_batches, batch_size, lr, head_size)  #TODO: remove unused vars

    model = EmbClassifier(encoder_from_pretrain, 512, num_classes, freeze_encoder, head_size=head_size)
    fps = [f for f in os.listdir(training_dir) if f.startswith("model_") and f.endswith(".pkl")]
    if fps:
        ckpts = [f.split("_")[1].split(".")[0] for f in fps]
        max_ckpt = max([eval(c) for c in ckpts])
        p = f"model_{max_ckpt}.pkl"
        print(f"######## found checkpoint {p}. loading its weights  ########")
        model.load_classify_from_pretrain(os.path.join(training_dir, p))

    # if encoder_from_pretrain is not None:
    #     assert os.path.exists(encoder_from_pretrain), encoder_from_pretrain
    #     if os.path.isdir(encoder_from_pretrain):
    #         encoder_from_pretrain = os.path.join(encoder_from_pretrain, "pytorch_model.bin")
    #     model.load_encoder_from_pretrain(encoder_from_pretrain)
    #     print(f"loaded encoder from {encoder_from_pretrain}")
    model.train()
    model = model.cuda()

    feature_extractor = RtfFeatureExtractor(feature_size = -1, sampling_rate=8000, padding_value = 0, process_time_domain=process_time_domain)
    audio_feature_extractor = None
    if not skip_audio:
        audio_feature_extractor = CarirFeatureExtractor(**json.load(open(os.path.join(encoder_from_pretrain, "preprocessor_config.json"), "r")))
    nsample_rir = (config.in_dim - 1) * 2 if not process_time_domain else config.in_dim
    collate_fn_p = collate_fn(feature_extractor=feature_extractor,
                              audio_feature_extractor=audio_feature_extractor,
                              nsample_rir=nsample_rir)
    if dummy_batches > 0:
        dataset = DummyDataset(dataset, dummy_batches)
    dataloader = DataLoader(dataset,
                            shuffle=True,
                            num_workers = num_workers,
                            batch_size = batch_size,
                            collate_fn=collate_fn_p,
                            drop_last=True,
                            worker_init_fn=dataset.worker_init_function)
    eval_dataloader = DataLoader(eval_dataset,
                            shuffle=True,
                            num_workers = num_workers,
                            batch_size = batch_size,
                            collate_fn=collate_fn_p,
                            drop_last=True,
                            worker_init_fn=dataset.worker_init_function)


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(training_dir, f'runs/{timestamp}'))
    model_path = os.path.join(training_dir, "model_{}.pkl")
    json.dump(config.__dict__, open(os.path.join(training_dir, "config.conf"), "w"))

    if clasify_target in ["room", "room_v4"]:
        rooms = {i: room for i, room in enumerate(eval_dataset.rooms)}
        pkl.dump(rooms, open(os.path.join(training_dir, "rooms.pkl"), "wb"))

    step = 0
    train_svm = False
    if train_svm:
        all_features = []
        all_gt = []
    for epoch in range(epochs):
        for rir_b in dataloader:
            if train_svm:
                if len(all_gt) > 100:
                    from .train_svm_classifier import train_svm, evaluate_svm
                    all_features_ = np.vstack(all_features)
                    all_gt_ = np.hstack(all_gt)
                    svm = train_svm(all_features_, all_gt_)
                    evaluate_svm(svm, all_features_, all_gt_)
                    break
            optimizer.zero_grad()

            model_input = rir_b["input_rtf_features"] if skip_audio else rir_b["input_audio_features"]
            model_input = model_input.cuda()
            rir_b["gt"] = rir_b["gt"].cuda()

            with torch.set_grad_enabled(not train_svm):
                if skip_audio:
                    scores = model(input_rtf_features=model_input)
                    if train_svm:
                        embeds =  model.get_rtf_emb(model_input)
                else:
                    scores = model(input_audio_features=model_input)
                    if train_svm:
                        embeds = model.get_audio_emb(model_input)
            if train_svm:
                all_gt.append(rir_b["gt"].cpu().numpy())
                all_features.append(embeds.cpu().numpy())
                continue

            if multi_class:
                loss, acc_x, acc_y, acc_z, argmin = calc_multi_class_loss(loss_fn, model_output, rir_b['gt'],
                                                                          num_classes,
                                                                          regression=multi_class_regression)
            else:
                loss = loss_fn(scores, rir_b["gt"])
                if step % acc_every == 0:
                    rank, acc, acc_by_class, num_samples = calc_acc(scores, rir_b["gt"], num_classes)

            loss.backward()
            optimizer.step()

            loss_f = loss.detach().cpu().numpy().item()
            writer.add_scalar('train/train_loss', loss_f, step)
            if multi_class:
                writer.add_scalar(f'train/argmin', argmin.sum() / batch_size, step)
                if multi_class_regression:
                    delta_x_std = acc_x.std()
                    delta_y_std = acc_y.std()
                    delta_z_std = acc_z.std()
                    acc_x = acc_x.mean()
                    acc_y = acc_y.mean()
                    acc_z = acc_z.mean()
                    writer.add_scalar(f'train/dist_x', acc_x, step)
                    writer.add_scalar(f'train/dist_y', acc_y, step)
                    writer.add_scalar(f'train/dist_z', acc_z, step)
                    writer.add_scalar(f'train/dist_x_std', delta_x_std, step)
                    writer.add_scalar(f'train/dist_y_std', delta_y_std, step)
                    writer.add_scalar(f'train/dist_z_std', delta_z_std, step)
                    print(f"in step {step} loss is: {loss_f:0.3f}: x {acc_x:0.3f}, y {acc_y:0.3f}, z {acc_z:0.3f}"
                          f" x_std {delta_x_std:0.3f}, y_std {delta_y_std:0.3f}, z_std {delta_z_std:0.3f}")
                else:
                    writer.add_scalar('train/accuracy', (acc_x + acc_y + acc_z)/3, step)
                    writer.add_scalar(f'train/acc_x', acc_x, step)
                    writer.add_scalar(f'train/acc_y', acc_y, step)
                    writer.add_scalar(f'train/acc_z', acc_z, step)
                    print(f"in step {step} loss is: {loss_f} acc_x is {acc_x:0.3f} acc_y is {acc_y:0.3f} "
                          f"acc_z is {acc_z:0.3f}, argmin: {argmin.sum() / batch_size}")
            else:
                if step % acc_every == 0:
                    rank, acc, acc_by_class, num_samples = calc_acc(scores, rir_b["gt"], num_classes)
                    writer.add_scalar('train/accuracy', acc, step)
                    writer.add_scalar('train/rank', rank, step)
                    for i in range(num_classes):
                        writer.add_scalar(f'train/num_samples_in_class_{i}', num_samples[i], step)
                        writer.add_scalar(f'train/acc_of_class_{i}', acc_by_class[i], step)
                    print(f"in step {step} loss is: {loss_f} acc is {acc} rank is {rank}")
                else:
                    print(f"in step {step} loss is: {loss_f}")
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            writer.flush()

            step = step + 1
        scheduler.step()

        # eval
        model.eval()
        with torch.no_grad():
            total_eval_loss = 0
            num_batches = 0
            total_acc = 0
            total_rank = 0
            total_num_samples = 0
            total_acc_by_class = {i: [] if multi_class else 0 for i in range(total_classes)}
            total_num_samples_of_class = {i: 0 for i in range(total_classes)}
            total_argmin = []

            for rir_b in eval_dataloader:
                num_batches = num_batches + 1

                model_input = rir_b["input_rtf_features"] if skip_audio else rir_b["input_audio_features"]
                model_input = model_input.cuda()
                rir_b["gt"] = rir_b["gt"].cuda()

                scores = model(input_rtf_features=model_input) if skip_audio else model(
                    input_audio_features=model_input)

                if multi_class:
                    total_num_samples = total_num_samples + model_output[0].shape[0]
                    eval_loss, acc_x, acc_y, acc_z, argmin = calc_multi_class_loss(loss_fn, model_output, rir_b['gt'],
                                                                                   num_classes,
                                                                                   regression=multi_class_regression)
                    total_argmin.append(argmin)
                    total_acc_by_class[0].append(acc_x)
                    total_acc_by_class[1].append(acc_y)
                    total_acc_by_class[2].append(acc_z)
                else:
                    eval_loss = loss_fn(scores, rir_b["gt"])
                    rank, acc, acc_by_class, num_samples_of_class = calc_acc(scores, rir_b["gt"], num_classes,
                                                                       normalize=False)
                    total_num_samples = total_num_samples + scores.shape[0]
                    total_acc = total_acc + acc
                    total_rank = total_rank + rank
                    for i in range(num_classes):
                        total_acc_by_class[i] = total_acc_by_class[i] + acc_by_class[i]
                        total_num_samples_of_class[i] = total_num_samples_of_class[i] + num_samples_of_class[i]

                total_eval_loss = total_eval_loss + eval_loss

        if multi_class:
            writer.add_scalar(f'eval/argmin', sum(a.sum() for a in argmin) / total_num_samples, step)
            if multi_class_regression:
                delta_x = torch.stack(total_acc_by_class[0])
                delta_y = torch.stack(total_acc_by_class[1])
                delta_z = torch.stack(total_acc_by_class[2])
                delta_x_std = delta_x.std()
                delta_y_std = delta_y.std()
                delta_z_std = delta_z.std()
                delta_x = delta_x.mean()
                delta_y = delta_y.mean()
                delta_z = delta_z.mean()
                writer.add_scalar(f'eval/dist_x', delta_x, step)
                writer.add_scalar(f'eval/dist_y', delta_y, step)
                writer.add_scalar(f'eval/dist_z', delta_z, step)
                writer.add_scalar(f'eval/dist_x_std', delta_x_std, step)
                writer.add_scalar(f'eval/dist_y_std', delta_y_std, step)
                writer.add_scalar(f'eval/dist_z_std', delta_z_std, step)
                print(f"\n\nin step {step} evaluation loss is: {loss_f:0.3f}: x {delta_x:0.3f}, y {delta_y:0.3f}, z {delta_z:0.3f}"
                      f" x_std {delta_x_std:0.3f}, y_std {delta_y_std:0.3f}, z_std {delta_z_std:0.3f}")
            else:
                acc_x = sum(total_acc_by_class[0]) / len(total_acc_by_class[0])
                acc_y = sum(total_acc_by_class[1]) / len(total_acc_by_class[1])
                acc_z = sum(total_acc_by_class[2]) / len(total_acc_by_class[2])
                writer.add_scalar('eval/accuracy', (acc_x + acc_y + acc_z) / 3, step)
                writer.add_scalar(f'eval/acc_x', acc_x, step)
                writer.add_scalar(f'eval/acc_y', acc_y, step)
                writer.add_scalar(f'eval/acc_z', acc_z, step)
                print(f"\n\nin step {step} evaluation loss is: {loss_f} acc_x is {acc_x:0.3f} acc_y is {acc_y:0.3f} "
                      f"acc_z is {acc_z:0.3f}, argmin: {argmin.sum() / batch_size}")
            print(f"in step {step} evaluation loss is: {total_eval_loss / num_batches} \n\n")
        else:
            print(f"\n\nin step {step} evaluation loss is: {total_eval_loss / num_batches} "
                  f"eval acc is { total_acc / total_num_samples} total_rank is {total_rank / num_batches}\n\n")
            writer.add_scalar('eval/eval_acc', total_acc / total_num_samples, step)
            writer.add_scalar('eval/total_rank', total_rank / num_batches, step)
            for i in range(num_classes):
                writer.add_scalar(f'eval/num_samples_in_class_{i}', total_num_samples_of_class[i], step)
                writer.add_scalar(f'eval/acc_of_class_{i}', total_acc_by_class[i] / total_num_samples_of_class[i], step)
        writer.add_scalar('eval/eval_loss', total_eval_loss / num_batches, step)
        writer.add_scalar('epoch', epoch, step)


        model.train()
        print(f"saving model in epoch {epoch}")
        torch.save(model.state_dict(), model_path.format(epoch+1))


import fire
if __name__ == '__main__':
    fire.Fire()