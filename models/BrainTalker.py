#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import fairseq
from models.FFT_block import Decoder


class GRU(nn.Module):
    def __init__(self, input_size=41, hidden_size=500, 
                  num_layers=3, output_size=500, 
                  batch_first=True, bidirectional=True):
        super(GRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first 
        self.bidirectional = bidirectional

        # model selection
        self.rnn = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=self.bidirectional, dropout=0.7)
        self.fc  = nn.Linear(2 * self.hidden_size, self.output_size)
    def forward(self, x, x_length):

        batch_size = x.size(0)
        x_length = x_length.detach().cpu()
        x = x.permute(0,2,1)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=self.batch_first)
        out, (hn1) = self.rnn(x)
        
        out, out_length = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)
        emb = self.fc(out)
        
        return emb.permute(1, 0, 2)


    def get_embedding(self, x, x_length):
        return self.forward(x, x_length)

class ConvTransposeBlock(nn.Module):
    def __init__(self, input_size, output_size, stride):
        super(ConvTransposeBlock, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.stride = stride

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.conv1d     = nn.ConvTranspose1d(self.input_size, 
                                             self.output_size, 3, stride=self.stride)
    def forward(self, x):
        
        x = self.leaky_relu(x)
        x = self.conv1d(x)
        return x


class MainModel(nn.Module):
    def __init__(self, device, channel, nmel):
        super(MainModel, self).__init__()

        cp_path = 'wav2vec_small.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device =  device
        self.model.eval()

        self.channel = channel
        self.nmel = nmel
        self.instancenorm  = nn.InstanceNorm1d(self.channel)
        self.make_one_ch   = nn.Linear(channel, 1)
        self.conv1d = ConvTransposeBlock(512, 128, 2)
        self.conv2d = ConvTransposeBlock(128, 128, 1)

        self.fc = nn.Linear(128, self.nmel)

        in_channel = 128
        d_model = 128
        d_inner = 512
        n_head = 2
        n_layers = 8
        fft_conv1d_kernel = (9, 1)
        fft_conv1d_padding = (4, 0)
        dropout = 0.3

        self.attention = Decoder(
                 in_channel,
                 d_model,
                 d_inner,
                 n_head,
                 n_layers,
                 fft_conv1d_kernel,
                 fft_conv1d_padding,             
                 dropout)
        self.rnn = GRU(input_size=512, hidden_size=64, num_layers=2, output_size=512)

    def forward(self, x, wav):
        with torch.no_grad():
            x = self.instancenorm(x)
            x = self.wav2vec(x.squeeze()) 

            wav = self.wav2vec(wav) 

        x = x.permute(2,1,0)     
        x = self.make_one_ch(x)  

        x_out = x.permute(2,1,0) 
        x_gru = self.rnn(x_out, torch.FloatTensor([x_out.size(2)])) 
        x_out = x_out + x_gru.permute(1,2,0)

        x = self.conv1d(x_out)
        x = self.conv2d(x)

        x = x.permute(0,2,1)
        x = self.attention(x)

        x = self.fc(x)
        x = x.permute(0,2,1)

        return x, x_out, wav

    def wav2vec(self, signal):
        rep = self.model.feature_extractor(signal)
        return rep
    
    def resampling(self, signal):
        signal = signal.cuda(self.device)
        rep = self.model.feature_extractor(signal)
        return rep