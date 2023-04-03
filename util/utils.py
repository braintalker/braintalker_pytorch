import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt

def accuracy(out, label):

    """
    out: 128, 1
    label: 128, 1
    """
    #residual = 1 - out
    #final = torch.cat([out, residual], 1)
    #pred = torch.argmin(final, dim=1)
    #corr_sum = torch.sum(pred.squeeze() == label.squeeze())

    pred  = torch.argmax(out, dim=1)
    target = label
    #target = torch.argmin(label, dim=1)

    corr_sum = torch.sum(pred.squeeze() == target.squeeze())
    print(corr_sum)
    return corr_sum

def get_writer(output_directory, log_directory):

    logging_path=f'{output_directory}/{log_directory}'
    if os.path.exists(logging_path) == False:
        os.makedirs(logging_path)
    writer = CustomWriter(logging_path)
            
    return writer

def plot_melspec(target, melspec, mel_lengths):

    fig, axes = plt.subplots(2, 1, figsize=(20,30))
    T = mel_lengths


    axes[0].imshow(target[:,:T],
                   origin='lower',
                   aspect='auto',
                   cmap='jet')

    axes[1].imshow(melspec[:,:T],
                   origin='lower',
                   aspect='auto',
                   cmap='jet')

    return fig


class CustomWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(CustomWriter, self).__init__(log_dir)
        
    def add_losses(self, name, phase, loss, global_step):
        self.add_scalar(f'{name}/{phase}', loss, global_step)
        
    def add_specs(self, name, phase, mel_padded, mel_out, mel_lengths, global_step):
        mel_fig = plot_melspec(mel_padded, mel_out, mel_lengths)    
        self.add_figure(f'melspec/{phase}', mel_fig, global_step)


def out_feature(i, k, s, p):
    o = math.floor((i + 2*p - k) / s) + 1
    return o

def final_out_feature(i, list_k, list_s, list_p):
    o = i
    for i in range(len(list_k)):
        k = list_k[i]
        s = list_s[i]
        p = list_p[i]
        o = out_feature(o, k, s, p)
    return o

def receptive_field(list_k, list_s):
    for i in range(len(list_k)):
        if i == 0:
            r = 1
        else:
            m_s = 1
            for j in range(i):
                m_s = m_s * list_s[j]
            r += (list_k[i] - 1) * m_s
    return r

def rf_fo(i, k, s, p):
    r = receptive_field(k, s)
    o = final_out_feature(i, k, s, p)
    return r, o


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, f'{filepath}/checkpoint_{iteration}')
    
def sqrt(inp):
    result = inp/2
    for i in range(30):
        result = (result + (inp / result)) / 2
    return result
def mean(inp):
    result = 0
    len_inp = len(inp)    
    for i in inp:
        result += i
    result = result / len_inp
    return result
def var(inp):
    result = 0
    len_inp = len(inp)
    for i in inp:
        result += (i - mean(inp)) ** 2
    result = result / len_inp
    return result
def std(inp):
    result = sqrt(var(inp))
    return result

def ci95(inp):
    max95 = mean(inp) + (1.96 * (std(inp) / sqrt(len(inp))))
    min95 = mean(inp) - (1.96 * (std(inp) / sqrt(len(inp))))
    return min95, max95