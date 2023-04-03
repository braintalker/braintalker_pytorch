import json
import os
import torch
import torch.nn as nn
from models.BrainTalker import MainModel
from data_utils import ECoGAudioDataset
from torch.utils.data import DataLoader
from util.utils import accuracy
import argparse
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torch
from util.utils import get_writer, save_checkpoint
import shutil


def trainer(config, config_path):

    result_folder     = config["result_folder"]
    experiment_folder = config["experiment_folder"] 
    gpu               = config["gpu"]
    device            = torch.device(f'cuda:{gpu}')
    writer            = get_writer(result_folder, experiment_folder) # Create a directory to store results
    shutil.copyfile(config_path, os.path.join(result_folder, experiment_folder, config_path)) # Save config file
    print(f'|-Training-| GPU:{gpu}')

    nmel = int(config["nmel"])
    channel = int(config["channel"])
    model = MainModel(device, channel, nmel)
    model.cuda(device)
    criterion = nn.MSELoss()

    learning_rate = config["learning_rate"]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 betas=(0.9, 0.98),
                                 eps=1e-09)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)


    trainset = ECoGAudioDataset(config, 'train')
    valset   = ECoGAudioDataset(config, 'val')
    train_loader = DataLoader(trainset,
                            shuffle=True,
                            batch_size=config["batch_size"], 
                            drop_last=True)
    val_loader = DataLoader(valset,
                            shuffle=False,
                            batch_size=1, 
                            drop_last=True)

    n_epoch          = config["n_epoch"]
    iteration        = config["iteration"]
    train_iteration  = config["train_iteration"]
    val_iteration    = config["val_iteration"]
    save_iteration   = config["save_iteration"]
    alpha = config["alpha"]
    beta  = config["beta"]

    count = 0
    for epoch in range(n_epoch):
        if count >= iteration:
            break

        l_tot = 0
        l_mel = 0 
        l_c   = 0 

        for x, label, wav in train_loader:

            count += 1
            x = x.cuda(device)
            wav = wav.cuda(device)

            label = label.type(torch.FloatTensor) 
            label = label.cuda(device)

            out, out_c, wav_c = model(x, wav)

            mel_loss  = criterion(out[:,:,:label.shape[2]], label) # mel loss
            c_loss    = criterion(out_c, wav_c) # context loss
            tot_loss  = mel_loss * alpha + c_loss * beta

            tot_loss.backward()

            l_tot  += tot_loss.item()
            l_mel  += mel_loss.item()
            l_c    += c_loss.item()

            optimizer.step()
            model.zero_grad()

        l_tot  = l_tot/ len(train_loader)
        l_mel  = l_mel/ len(train_loader)
        l_c    = l_c/ len(train_loader)

        if epoch % train_iteration == 0:
            print(f'|-Training-| Epoch {epoch}: {l_tot:.3f}')
            writer.add_losses("Train", "tot", l_tot, epoch)
            writer.add_losses("Train", "mel loss", l_mel, epoch)
            writer.add_losses("Train", "context loss", l_c, epoch)
            writer.add_specs("Mel", "train", label.squeeze().detach().cpu().numpy(), out.squeeze().detach().cpu().numpy(), label.shape[-1], epoch) 


        if epoch % val_iteration == 0:
            l_tot = 0
            l_mel = 0
            l_c   = 0
            model.eval()
            with torch.no_grad():
                for x, label, wav in val_loader:

                    x = x.cuda(device)
                    wav = wav.cuda(device)
                    label = label.type(torch.FloatTensor) 
                    label = label.cuda(device)

                    out, out_c, wav_c = model(x, wav)

                    mel_loss  = criterion(out[:,:,:label.shape[2]], label)
                    c_loss    = criterion(out_c, wav_c)
                    tot_loss  = mel_loss * alpha + c_loss * beta
                    
                    l_tot  += tot_loss.item()
                    l_mel  += mel_loss.item()
                    l_c    += c_loss.item()

                l_tot  = l_tot/ len(val_loader)
                l_mel  = l_mel/ len(val_loader)
                l_c    = l_c/ len(val_loader)

                print(f'|-Validation-|{l_tot :.3f}')
                writer.add_losses("Validation", "tot", l_tot, epoch)
                writer.add_losses("Validation", "mel loss", l_mel, epoch)
                writer.add_losses("Validation", "context loss", l_c, epoch) 
                writer.add_specs("Mel", "val", label.squeeze().detach().cpu().numpy(), out.squeeze().detach().cpu().numpy(), label.shape[-1], epoch)     
        if epoch % save_iteration == 0:
            save_checkpoint(model, optimizer, learning_rate, epoch, f'{result_folder}/{experiment_folder}')
        scheduler.step()
        model.train()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main')    

    parser.add_argument('experiment_folder', help='write down experiment name')
    parser.add_argument('gpu', type=int, help='gpu index which you want to use')

    args = parser.parse_args() 

    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    config_path = os.path.join(task_folder, 'config.json')
    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    config["experiment_folder"] = args.experiment_folder
    config["gpu"] = args.gpu
    trainer(config, config_path)