import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.helpers.utils import (signal_noise_ratio as snr, 
                               scale_invariant_signal_noise_ratio as si_snr,
                               phase_constrained_magnitude as pcm,
                               delta_ILD as ild, delta_IPD as ipd, delta_ITD_cc as itd_cc, delta_ITD_gccphat as itd_gccphat)

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func



class Net(nn.Module):
    def __init__(self, n_srcs=4, win=512, n_mics=4, n_layers=6, 
                 att_dim=64, hidden_dim=256, n_head=4, emb_dim=64, 
                 emb_ks=4, emb_hs=1, dropout=0.1, eps=1.0e-5,
                 clue_type='time-azim', label_len=41, angle_dim=40):
        super(Net, self).__init__()
        
        self.n_srcs = n_srcs
        self.win = win
        self.hop = win // 2
        self.n_mics = n_mics
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.clue_type = clue_type # 'label', 'time-label', 'azim', 'time-azim'
        assert win % 2 == 0

        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        
        self.conv = nn.Sequential(
            nn.Conv2d(2 * n_mics, emb_dim * n_head, ks, padding=padding),
            nn.GroupNorm(1, emb_dim * n_head, eps=eps),
            InverseDenseBlock2d(emb_dim * n_head, emb_dim, n_head)
        )
        
        self.mix_blocks = nn.ModuleList([])
        for idx in range(self.n_layers):
            self.mix_blocks.append(DeFTANblock(idx, emb_dim, emb_ks, emb_hs, att_dim, hidden_dim, n_head, dropout, eps))
            
        # clue encoding
        if clue_type == 'label' or clue_type == 'time-label':
            self.label_embedding = nn.ModuleList([])
            for idx in range(self.n_layers):
                self.label_embedding.append(nn.Sequential(
                    nn.Linear(label_len, emb_dim),
                    nn.LayerNorm(emb_dim),
                    nn.PReLU()
                    ))

        elif clue_type == 'azim' or clue_type == 'time-azim':
            self.azim_embedding = nn.ModuleList([])
            for idx in range(self.n_layers):
                self.azim_embedding.append(nn.Sequential(
                    nn.Linear(angle_dim, emb_dim),
                    nn.LayerNorm(emb_dim),
                    nn.PReLU()
                    ))

        self.deconv = nn.Sequential(
            nn.Conv2d(emb_dim, 2 * n_srcs * n_head, ks, padding=padding),
            InverseDenseBlock2d(2 * n_srcs * n_head, 2 * n_srcs, n_head)
        )

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

    def forward(self, input, gt, label_vec, time_label_vec, azim_vec, time_azim_vec):
        _input = input
        input, rest = self.pad_signal(input)
        B, M, N = input.size()  # B: # of batches, M: # of mics, N: # of samples
        mix_std_ = torch.std(input, dim=(1, 2), keepdim=True)  # [B, 1, 1]
        input = input / mix_std_  # RMS normalization

        stft_input = torch.stft(input.view([-1, N]), n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(input.type()), return_complex=False)
        _, F, T, _ = stft_input.size()               # B*M , F: # of freq bins, T: # of frames, 2: real & imag
        xi = stft_input.view([B, M, F, T, 2])        # B*M, F, T, 2 -> B, M, F, T, 2
        xi = xi.permute(0, 1, 4, 3, 2).contiguous()  # [B, M, 2, T, F]
        batch = xi.view([B, M * 2, T, F])            # [B, 2*M, T, F]

        batch = self.conv(batch)                     # [B, C, T, F]

        for ii in range(self.n_layers):
            if self.clue_type == 'label':
                batch = batch * rearrange(self.label_embedding[ii](label_vec), 'b c -> b c 1 1')         # [B, C, T, F]
            elif self.clue_type == 'time-label':
                batch = batch * rearrange(self.label_embedding[ii](time_label_vec), 'b t c -> b c t 1')  # [B, C, T, F]
            elif self.clue_type == 'azim':
                batch = batch * rearrange(self.azim_embedding[ii](azim_vec), 'b c -> b c 1 1')           # [B, C, T, F]
            elif self.clue_type == 'time-azim':
                batch = batch * rearrange(self.azim_embedding[ii](time_azim_vec), 'b t c -> b c t 1')    # [B, C, T, F]

            batch = self.mix_blocks[ii](batch)       # [B, C, T, F]

        batch = self.deconv(batch).view([B, self.n_srcs, 2, T, F]).view([B * self.n_srcs, 2, T, F])

        batch = batch.permute(0, 3, 2, 1).type(input.type())                # [B*n_srcs, 2, T, F] -> [B*n_srcs, F, T, 2]
        istft_input = torch.complex(batch[:, :, :, 0], batch[:, :, :, 1])
        istft_output = torch.istft(istft_input, n_fft=self.win, hop_length=self.hop, window=torch.hann_window(self.win).type(input.type()), return_complex=False)

        output = istft_output[:, self.hop:-(rest + self.hop)].unsqueeze(1)  # [B*n_srcs, 1, N]
        output = output.view([B, self.n_srcs, -1])                          # [B, n_srcs, N]
        output = output * mix_std_  # reverse the RMS normalization

        return output, loss(_input, output, gt, self.win, self.hop)


class InverseDenseBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(out_channels * ((idx > 0) + 1), out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1, out_channels, 1e-5),
                nn.PReLU(out_channels)
            ))

    def forward(self, x):
        B, C, L = x.size()
        g = self.groups
        x = x.view(B, g, C//g, L).transpose(1, 2).reshape(B, C, L) ###
        skip = x[:, ::g, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :]], dim=1)
        return output


class InverseDenseBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        assert in_channels // out_channels == groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.blocks = nn.ModuleList([])
        for idx in range(groups):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(out_channels * ((idx > 0) + 1), out_channels, kernel_size=(3, 3), padding=(1, 1)),
                nn.GroupNorm(1, out_channels, 1e-5),
                nn.PReLU(out_channels)
            ))

    def forward(self, x):
        B, C, T, Q = x.size()
        g = self.groups
        x = x.view(B, g, C//g, T, Q).transpose(1, 2).reshape(B, C, T, Q) ###
        skip = x[:, ::g, :, :]
        for idx in range(g):
            output = self.blocks[idx](skip)
            skip = torch.cat([output, x[:, idx+1::g, :, :]], dim=1)
        return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.cv_qk = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=3, padding=1, bias=False),
            nn.GLU(dim=1))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.p_drop = dropout

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qk = self.cv_qk(x.transpose(1, 2)).transpose(1, 2)
        q = rearrange(self.to_q(qk), 'b n (h d) -> b n h d', h=self.heads)
        q = torch.tensor(q, dtype=torch.float16).to(x.device)
        k = rearrange(self.to_k(qk), 'b n (h d) -> b n h d', h=self.heads)
        k = torch.tensor(k, dtype=torch.float16).to(x.device)
        v = rearrange(self.to_v(x), 'b n (h d) -> b n h d', h=self.heads)
        v = torch.tensor(v, dtype=torch.float16).to(x.device)

        out = flash_attn_func(q, k, v, dropout_p=self.p_drop, softmax_scale=self.scale)
        out = torch.tensor(out, dtype=torch.float32).to(x.device)
        
        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, idx, dropout):
        super().__init__()
        self.PW1 = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.GELU(),
	        nn.Dropout(dropout)
        )
        self.PW2 = nn.Sequential(
            nn.Linear(dim, hidden_dim//2),
            nn.GELU(),
	        nn.Dropout(dropout)
        )
        self.DW_Conv = nn.Sequential(
            nn.Conv1d(hidden_dim//2, hidden_dim//2, kernel_size=5, dilation=2**idx, padding='same'),
            nn.GroupNorm(1, hidden_dim//2, 1e-5),
            nn.PReLU(hidden_dim//2)
        )
        self.PW3 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        ffw_out = self.PW1(x)
        dw_out = self.DW_Conv(self.PW2(x).transpose(1, 2)).transpose(1, 2)
        out = self.PW3(torch.cat((ffw_out, dw_out), dim=2))
        return out


class DeFTANblock(nn.Module):
    def __getitem__(self, key):
        return getattr(self, key)

    def __init__(self, idx, emb_dim, emb_ks, emb_hs, att_dim, hidden_dim, n_head, dropout, eps):
        super().__init__()
        in_channels = emb_dim * emb_ks
        self.intra_norm = LayerNormalization4D(emb_dim, eps)
        self.intra_inv = InverseDenseBlock1d(in_channels, emb_dim, emb_ks)
        self.intra_att = PreNorm(emb_dim, Attention(emb_dim, n_head, att_dim, dropout))
        self.intra_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim, idx, dropout))
        self.intra_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.inter_norm = LayerNormalization4D(emb_dim, eps)
        self.inter_inv = InverseDenseBlock1d(in_channels, emb_dim, emb_ks)
        self.inter_att = PreNorm(emb_dim, Attention(emb_dim, n_head, att_dim, dropout))
        self.inter_ffw = PreNorm(emb_dim, FeedForward(emb_dim, hidden_dim, idx, dropout))
        self.inter_linear = nn.ConvTranspose1d(emb_dim, emb_dim, emb_ks, stride=emb_hs)

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x):
        B, C, old_T, old_Q = x.shape
        T = math.ceil((old_T - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        Q = math.ceil((old_Q - self.emb_ks) / self.emb_hs) * self.emb_hs + self.emb_ks
        x = F.pad(x, (0, Q - old_Q, 0, T - old_T))

        # F-transformer
        input_ = x
        intra_rnn = self.intra_norm(input_)                                                    # [B, C, T, Q]
        intra_rnn = intra_rnn.transpose(1, 2).contiguous().view(B * T, C, Q)                   # [BT, C, Q]
        intra_rnn = F.unfold(intra_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BT, C*emb_ks, -1]
        intra_rnn = self.intra_inv(intra_rnn)                                                  # [BT, C, -1]

        intra_rnn = intra_rnn.transpose(1, 2)                                                  # [BT, -1, C]
        intra_rnn = self.intra_att(intra_rnn) + intra_rnn
        intra_rnn = self.intra_ffw(intra_rnn) + intra_rnn
        intra_rnn = intra_rnn.transpose(1, 2)                                                  # [BT, H, -1]

        intra_rnn = self.intra_linear(intra_rnn)                                               # [BT, C, Q]
        intra_rnn = intra_rnn.view([B, T, C, Q])
        intra_rnn = intra_rnn.transpose(1, 2).contiguous()                                     # [B, C, T, Q]
        intra_rnn = intra_rnn + input_                                                         # [B, C, T, Q]

        # T-transformer
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)                                                    # [B, C, T, F]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous().view(B * Q, C, T)               # [BF, C, T]
        inter_rnn = F.unfold(inter_rnn[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))  # [BF, C*emb_ks, -1]
        inter_rnn = self.inter_inv(inter_rnn)                                                  # [BF, C, -1]

        inter_rnn = inter_rnn.transpose(1, 2)                                                  # [BF, -1, C]
        inter_rnn = self.inter_att(inter_rnn) + inter_rnn
        inter_rnn = self.inter_ffw(inter_rnn) + inter_rnn
        inter_rnn = inter_rnn.transpose(1, 2)                                                  # [BF, H, -1]

        inter_rnn = self.inter_linear(inter_rnn)                                               # [BF, C, T]
        inter_rnn = inter_rnn.view([B, Q, C, T])
        inter_rnn = inter_rnn.permute(0, 2, 3, 1).contiguous()                                 # [B, C, T, Q]
        inter_rnn = inter_rnn + input_                                                         # [B, C, T, Q]

        return inter_rnn


class LayerNormalization4D(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            _, C, _, _ = x.shape
            stat_dim = (1,)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class LayerNormalization4DCF(nn.Module):
    def __init__(self, input_dimension, eps=1e-5):
        super().__init__()
        assert len(input_dimension) == 2
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x):
        if x.ndim == 4:
            stat_dim = (1, 3)
        else:
            raise ValueError("Expect x to have 4 dimensions, but got {}".format(x.ndim))
        mu_ = x.mean(dim=stat_dim, keepdim=True)  # [B,1,T,1]
        std_ = torch.sqrt(
            x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat
    
    
    
# Define optimizer, loss and metrics
def optimizer(model, data_parallel=False, **kwargs):
    import torch.optim as optim
    return optim.Adam(model.parameters(), **kwargs)

def loss(mix, pred, tgt, win, hop):
    pcm_loss = (pcm(mix, pred, tgt, win, hop)).mean()
    return pcm_loss
    
def metrics(mixed, output, gt):
    """ Function to compute metrics (on the total duration, 6-second) """
    metrics = {}

    def _metric(metric, pred, tgt):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t)).cpu().item())
        return _vals

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals
    
    def metric_ild(metric, pred, tgt, idx1, idx2):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t, idx1, idx2)).cpu().item())
        return _vals
    
    def metric_ipd(metric, pred, tgt, idx1, idx2, win=256, hop=128):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t, win, hop, idx1, idx2)).cpu().item())
        return _vals
    
    def metric_itd(metric, pred, tgt, idx1, idx2, fs=8000):
        _vals = []
        for t, p in zip(tgt, pred):
            _vals.append((metric(p, t, fs, idx1, idx2)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        ### Annotate the following codes when `evaluation` ###
        # metrics[m_fn.__name__] = metric_i(m_fn, mixed, output, gt)
        ### Annotate the following codes when `training` ###
        metrics[m_fn.__name__] = _metric(m_fn, output, gt) # SNR & SI-SNR
        metrics[m_fn.__name__+'_i'] = metric_i(m_fn, mixed, output, gt) # SNRi & SI-SNRi

    pair_list = ['_12', '_13', '_14', '_23', '_24', '_34']
    for pair in pair_list:
        idx1, idx2 = int(pair[1]) - 1, int(pair[2]) - 1
        metrics[ild.__name__+pair] = metric_ild(ild, output, gt, idx1, idx2)
        metrics[ipd.__name__+pair] = metric_ipd(ipd, output, gt, idx1, idx2, win=256, hop=128)
        ### Recommend to annotate the following code when `training` (take much time) ###
        # metrics[itd_cc.__name__+pair] = metric_itd(itd_cc, output, gt, idx1, idx2, fs=8000)
        metrics[itd_gccphat.__name__+pair] = metric_itd(itd_gccphat, output, gt, idx1, idx2, fs=8000)

    return metrics
