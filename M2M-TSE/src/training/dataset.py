"""
Torch dataset object for synthetically rendered spatial data.
"""

import os
import json
import jams
import math
import random
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scaper_edited as scaper
import torch
import torchaudio
import torchaudio.transforms as AT
from torch.autograd import Variable
from random import randrange



class FSDSoundScapesDataset(torch.utils.data.Dataset):
    """
    Base class for FSD Sound Scapes dataset
    """

    _labels = [
    "Acoustic_guitar", "Applause", "Bark", "Bass_drum",
    "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet",
    "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping",
    "Fireworks", "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire",
    "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow",
    "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone",
    "Trumpet", "Violin_or_fiddle", "Writing"]

    def __init__(self, input_dir, dset='', sr=8000, win=256,
                 resample_rate=None, max_num_targets=1, 
                 azim_type='cycpos', d_model=40, alpha=20, 
                 data_div_rate=4, ts_only=0):
        
        self.input_dir = input_dir
        assert dset in ['train', 'val', 'test'], \
            "`dset` must be one of ['train', 'val', 'test']"
        self.dset = dset
        self.sr = sr
        self.win = win; self.hop = win // 2
        self.max_num_targets = max_num_targets
        self.azim_type = azim_type
        self.d_model = d_model
        self.alpha = alpha
        self.ts_only = ts_only
        
        self.fg_dir = os.path.join(input_dir, 'FSDKaggle2018/%s' % dset)
        self.bg_dir = '../dataset'
        logging.info("Loading %s dataset: fg_dir=%s bg_dir=%s" %
                     (dset, self.fg_dir, self.bg_dir))
        
        self.samples = sorted(list(
            Path(os.path.join(input_dir, 'jams', dset)).glob('[0-9]*')))
        if dset in ['train', 'test']:
            self.samples = self.samples[:len(self.samples) // data_div_rate]

        jamsfile = os.path.join(self.samples[0], 'mixture.jams')
        _, _jams, _, _, _ = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        _sr = _jams['annotations'][0]['sandbox']['scaper']['sr']
        assert _sr == sr, "Sampling rate provided does not match the data"        

    def _get_label_vector(self, labels):
        """
        Generates a multi-hot vector corresponding to `labels`.
        """
        vector = torch.zeros(len(FSDSoundScapesDataset._labels))

        for label in labels:
            idx = FSDSoundScapesDataset._labels.index(label)
            assert vector[idx] == 0, "Repeated labels"
            vector[idx] = 1 

        return vector
    
    def _get_azim_angle_vector_onehot(self, angles, resolution=1):
        """
        Generates a multi-hot vector corresponding to `angles`.
        """
        num_positions = int(360 / resolution) # 144 if 2.5 degree resolution
        vector = torch.zeros(num_positions)

        for angle in angles:
            ### For checking the sensitivity to DoA mismatch
            # mismatch = 50 # (< 180 degrees)
            # positive = True # True or False, whether to add or subtract the mismatch
            # if positive:
            #     angle = (angle + mismatch) % 360
            # else:
            #     angle = angle - mismatch
            #     if angle < 0:
            #         angle += 360

            if round(angle / resolution) == num_positions:
                angle = 0
            index = int(round(angle / resolution))
            vector[index] = 1

        return vector
    
    def _get_azim_angle_vector_cycpos(self, angle):
        """
        Generates a cyclic positional encoding vector corresponding to azimuth angle of `labels`.
        """
        vector = torch.zeros(self.d_model)

        max_len = 360 # total 360 degree (1 degree resolution)
        d_model = self.d_model
        alpha = self.alpha

        # Compute the positional encodings in log space
        pe = torch.zeros(max_len, d_model)
        phi = torch.arange(0, max_len).unsqueeze(1) * (math.pi / 180)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))
        pe[:, 0::2] = torch.sin(torch.sin(phi) * alpha * div_term)
        pe[:, 1::2] = torch.sin(torch.cos(phi) * alpha * div_term)

        # Equalize L2-norm for all angles
        for a in range(max_len):
            pe[a] = pe[a] / torch.norm(pe[a])
        
        ### For checking the sensitivity to DoA mismatch
        # mismatch = 50 # (< 180 degrees)
        # positive = True # True or False, whether to add or subtract the mismatch
        # if positive:
        #     angle = (angle + mismatch) % 360
        # else:
        #     angle = angle - mismatch
        #     if angle < 0:
        #         angle += 360
        
        if round(angle) == 360:
            angle = 0
        vector = pe[round(angle)]

        return vector
    
    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nchannel = input.size(1)
        nsample = input.size(2)

        rest = self.win - (self.hop + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nchannel, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(batch_size, nchannel, self.hop)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
    
    def stft(self, input):
        input, rest = self.pad_signal(input)
        B, M, N = input.size()  # B: # of batches, M: # of mics, N: # of samples

        stft_input = torch.stft(input.view([-1, N]), n_fft=self.win, hop_length=self.hop, 
                                window=torch.hann_window(self.win).type(input.type()), 
                                return_complex=False)
        _, F, T, _ = stft_input.size() # B*M , F: # of freq bins, T: # of frames, 2: real & imag
        
        return stft_input

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        jamsfile = os.path.join(sample_path, 'mixture.jams')

        mixture, _jams, ann_list, event_audio_list, angle_list = scaper.generate_from_jams(
            jamsfile, fg_path=self.fg_dir, bg_path=self.bg_dir)
        # ann_list: [onset  offset  label]
        # event_audio_list: individual audio listed in order same with jams file
        # angle_list: (azimuth_angle, elevation_angle)

        isolated_events = {}
        isolated_angles = {}
        isolated_stft_durations = {}
        for e, a, angle in zip(ann_list, event_audio_list[1:], angle_list):
            # 0th event is background
            isolated_events[e[2]] = a
            isolated_angles[e[2]] = angle # (azimuth_angle, elevation_angle)
            t_start = int((self.sr * e[0]) // self.hop + 1) # + 1 for the padding in initial processing
            t_end = int((self.sr * e[1]) // self.hop + 1) # + 1 for the padding in initial processing
            isolated_stft_durations[e[2]] = (t_start, t_end)
        gt_events = list(pd.read_csv(
            os.path.join(sample_path, 'gt_events.csv'), sep='\t')['label'])

        mixture = torch.from_numpy(mixture).permute(1, 0).to(torch.float)
        T = self.stft(mixture.unsqueeze(0)).shape[2]

        if self.dset == 'train':
            labels = random.sample(gt_events, randrange(1, self.max_num_targets + 1))
        elif self.dset == 'val':
            labels = gt_events[:idx % self.max_num_targets + 1]
        elif self.dset == 'test':
            labels = gt_events[:self.max_num_targets]
            
        gt = torch.zeros_like(torch.from_numpy(event_audio_list[1]).permute(1, 0))
        for l in labels:
            gt = gt + torch.from_numpy(isolated_events[l]).permute(1, 0).to(torch.float)
            azim_angle = isolated_angles[l][0] # for single target only
        
        label_vec = self._get_label_vector(labels)

        num_labels = len(FSDSoundScapesDataset._labels)
        time_label_vec = torch.zeros(T, num_labels)
        for i in range(len(labels)):
            t_start = isolated_stft_durations[labels[i]][0]
            t_end = isolated_stft_durations[labels[i]][1]
            if self.ts_only == 0:
                time_label_vec[t_start:t_end] += self._get_label_vector([labels[i]])
            else:
                time_label_vec[t_start:t_end] += (torch.ones(num_labels) / torch.norm(torch.ones(num_labels)))
            
        if self.azim_type == 'onehot':
            azim_vec = self._get_azim_angle_vector_onehot([azim_angle], resolution=1)
        elif self.azim_type == 'cycpos':
            azim_vec = self._get_azim_angle_vector_cycpos(azim_angle)
        
        time_azim_vec = torch.zeros(T, self.d_model)
        for i in range(len(labels)):
            t_start = isolated_stft_durations[labels[i]][0]
            t_end = isolated_stft_durations[labels[i]][1]
            if self.ts_only == 0:
                if self.azim_type == 'onehot':
                    time_azim_vec[t_start:t_end] += self._get_azim_angle_vector_onehot([isolated_angles[labels[i]][0]], resolution=1)
                elif self.azim_type == 'cycpos':
                    time_azim_vec[t_start:t_end] += self._get_azim_angle_vector_cycpos(isolated_angles[labels[i]][0])
            else:
                time_azim_vec[t_start:t_end] += (torch.ones(self.d_model) / torch.norm(torch.ones(self.d_model)))

        return mixture, gt, label_vec, time_label_vec, azim_vec, time_azim_vec



def tensorboard_add_sample(writer, tag, sample, step, params):
    """
    Adds a sample of FSDSynthDataset to tensorboard.
    """
    if params['resample_rate'] is not None:
        sr = params['resample_rate']
    else:
        sr = params['sr']

    m, l, gt, o, tl = sample
    m, gt, o = m.cpu(), gt.cpu(), o.cpu()

    def _add_audio(a, audio_tag, axis, plt_title):
        for i, ch in enumerate(a):
            axis.plot(ch, label='mic %d' % i)
            writer.add_audio(
                '%s/mic %d' % (audio_tag, i), ch.type(torch.float64), step, sr)
        axis.set_title(plt_title)
        axis.legend()

    for b in range(m.shape[0]):
        label = []
        for i in range(len(l[b, :])):
            if l[b, i] == 1:
                label.append(FSDSoundScapesDataset._labels[i])

        # Add waveforms
        rows = 3 # input, output, gt
        fig = plt.figure(figsize=(10, 2 * rows))
        axes = fig.subplots(rows, 1, sharex=True)
        _add_audio(m[b], '%s/sample_%d/0_input' % (tag, b), axes[0], "Mixed")
        _add_audio(o[b], '%s/sample_%d/1_output' % (tag, b), axes[1], "Output (%s)" % label)
        _add_audio(gt[b], '%s/sample_%d/2_gt' % (tag, b), axes[2], "GT (%s)" % label)
        writer.add_figure('%s/sample_%d/waveform' % (tag, b), fig, step)

def tensorboard_add_metrics(writer, tag, metrics, label, step):
    """
    Add metrics to tensorboard.
    """
    vals = np.asarray(metrics['scale_invariant_signal_noise_ratio'], dtype='float') # _i_tot
    vals = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=-1.0)

    writer.add_histogram('%s/%s' % (tag, 'SI-SNRi'), vals, step)

    label_names = [FSDSoundScapesDataset._labels[torch.argmax(_)] for _ in label]
    for l, v in zip(label_names, vals):
        writer.add_histogram('%s/%s' % (tag, l), v, step)



def test_main():
    from tqdm import tqdm

    data_test = FSDSoundScapesDataset(input_dir='../dataset/FSDSoundScapes_MC_same_height_8k_noise', dset='test', sr=8000, win=256, 
                                      azim_type='cycpos', d_model=40, alpha=20, data_div_rate=4, ts_only=0)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False)

    for batch_idx, (mixed, gt, label_vec, time_label_vec, azim_vec, time_azim_vec) in enumerate(tqdm(test_loader)):

        print(mixed.shape, gt.shape) # torch.Size([1, 4, 48000]) torch.Size([1, 4, 48000])
        print(label_vec.shape, time_label_vec.shape) # torch.Size([1, 41]) torch.Size([1, 380, 41])
        print(azim_vec.shape, time_azim_vec.shape) # torch.Size([1, 40]) torch.Size([1, 380, 40])

        break

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    test_main()
