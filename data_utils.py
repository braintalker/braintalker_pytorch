import torch
import torch.utils.data
import torch.nn.functional as F
import os
import glob
import torch.nn as nn
import pickle
from scipy import signal

class ECoGAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, process_type):

        self.process_type = process_type
        if self.process_type == 'train':
            path_data = config["trainset_folder"]
        else:
            path_data = config["testset_folder"]
        self.data_folder = path_data
        self.files = [x for x in glob.glob(self.data_folder + '/ecog/*.pkl')]
        self.files = [x.split('/')[-1].replace('.pkl', '') for x in self.files]

        self.sr_ecog = config["sampling_rate_ecog"]
        self.sr_wav  = config["sampling_rate_wav"]
        self.gpu = config["gpu"]
        self.device = torch.device(f'cuda:{self.gpu}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        path_ecog = os.path.join(self.data_folder,'ecog', self.files[idx] + '.pkl')
        path_mel  = os.path.join(self.data_folder,'mel', self.files[idx]+ '.pkl')
        path_wav  = os.path.join(self.data_folder,'wav', self.files[idx]+ '.pkl')

        with open(path_ecog, 'rb') as f:
            feat = pickle.load(f)
        with open(path_mel, 'rb') as f:
            mel = pickle.load(f)
        with open(path_wav, 'rb') as f:
            wav = pickle.load(f)
        wav = torch.FloatTensor(wav)

        tmp = []
        for i in range(feat.shape[0]):
            f = signal.resample(feat[i], feat.shape[1] * 8)
            f = torch.FloatTensor(f)
            tmp.append(f.unsqueeze(0))
        tmp = torch.cat(tmp, 0)
        feat = tmp

        return feat, mel, wav
