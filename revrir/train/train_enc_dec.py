import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from datetime import datetime


from ..data.dummy_dataset import DummyDataset
from ..data.enc_dec_dataset import CARIR_AutoEncoder_dataset, collate_fn

from ..utils import get_generater_rirs_paths, makedirs_p
from ..feature_extraction_carir import RtfFeatureExtractor
from ..models.enc_dec_rtf import EncoderDecoder, RtfModelConfig



def train(num_workers=1,
          epochs=5,
          training_dir = '',
          loss = "L1",
          num_mid_layers = 1,
          freeze_encoder = True,
          encoder_from_pretrain = None,
          dummy_batches = -1,
          batch_size = 50,
          lr = -1,
          ):
    makedirs_p(training_dir)
    rir_paths, md_paths = get_generater_rirs_paths("train_30K")
    dataset = CARIR_AutoEncoder_dataset(rir_paths=rir_paths,
                                        md_paths=md_paths)
    eval_rir_paths, eval_md_paths = get_generater_rirs_paths("test")
    eval_dataset = CARIR_AutoEncoder_dataset(rir_paths=eval_rir_paths,
                                             md_paths=eval_md_paths)

    config = RtfModelConfig(2601, 768, 768, 512, "relu", loss, num_mid_layers,
                            freeze_encoder, encoder_from_pretrain, dummy_batches, batch_size, lr)

    model = EncoderDecoder(config)
    if encoder_from_pretrain is not None:
        assert os.path.exists(encoder_from_pretrain), encoder_from_pretrain
        if os.path.isdir(encoder_from_pretrain):
            encoder_from_pretrain = os.path.join(encoder_from_pretrain, "pytorch_model.bin")
        model.load_encoder_from_pretrain(encoder_from_pretrain)
        print(f"loaded encoder from {encoder_from_pretrain}")
    model.train()
    model = model.cuda()

    feature_extractor = RtfFeatureExtractor(feature_size = -1, sampling_rate=8000, padding_value = 0)
    nsample_rir = (config.in_dim - 1) * 2
    collate_fn_p = collate_fn(feature_extractor=feature_extractor,
                              nsample_rir=nsample_rir,
                              )
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

    if config.loss == "L1":
        print("using loss L1")
        loss_fn = nn.L1Loss(reduction='mean')
        if lr < 0:
            lr = 0.001
    elif config.loss == "MSE":
        print("using loss MSE")
        loss_fn = nn.MSELoss(reduction='mean')
        if lr < 0:
            lr = 0.0001
    config.lr = lr
    print(f"lr is: {lr}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=epochs)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs, power=0.1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(training_dir, f'runs/{timestamp}'))
    model_path = os.path.join(training_dir, "model_{}.pkl")
    json.dump(config.__dict__, open(os.path.join(training_dir, "config.conf"), "w"))

    step = 0
    for epoch in range(epochs):
        for rir_b in dataloader:
            optimizer.zero_grad()

            model_input = rir_b["input_rtf_features"]
            model_input = model_input.cuda()
            rtf_fft = model(model_input)

            loss = loss_fn(rtf_fft["rtf_features"], model_input)
            if torch.isnan(loss):
                print("loss is nan")
                exit()
            loss.backward()
            optimizer.step()

            loss_f = loss.detach().cpu().numpy().item()
            writer.add_scalar('train_loss', loss_f, step)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step)
            print(f"in step {step} loss is: {loss_f}")
            writer.flush()

            step = step + 1
        scheduler.step()

        # eval
        model.eval()
        with torch.no_grad():
            total_eval_loss = 0
            num_batches = 0
            for rir_b in eval_dataloader:
                num_batches = num_batches + 1
                model_input = rir_b["input_rtf_features"]
                model_input = model_input.cuda()
                rtf_fft = model(model_input)

                eval_loss = loss_fn(rtf_fft["rtf_features"], model_input)
                total_eval_loss = total_eval_loss + eval_loss

        print(f"\n\nin step {step} evaluation loss is: {total_eval_loss / num_batches}\n\n")
        writer.add_scalar('eval_loss', total_eval_loss / num_batches, step)
        writer.add_scalar('epoch', epoch, step)


        model.train()
        print(f"saving model in epoch {epoch}")
        torch.save(model.state_dict(), model_path.format(epoch+1))


import fire
if __name__ == '__main__':
    fire.Fire()