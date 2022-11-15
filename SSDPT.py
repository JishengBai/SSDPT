#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torch.autograd import Variable
import random

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

class DPTrans_encoder(nn.Module):
    ''' DPTrans encoder'''
    def __init__(self, frames, bins, \
                 nhead, dim_feedforward, n_layers, dropout):
        '''
        Parameters
        ----------
        
        frames : int, the number of reduced frames.
        bins : int, the number of reduced bins.
        
        nhead : int, number of heads in multiheadattention.
        dim_feedforward : int, the dimension of feedforward network.
        n_layers : int, the number of encoder layers.
        dropout : float, dropout value.
        
        reduce_type : str, optional 'max' or 'avg'.

        Returns
        -------
        None.

        '''
        super(DPTrans_encoder, self).__init__()
        
        self.encoder_layer_frames = nn.TransformerEncoderLayer(bins, nhead,\
                                                        dim_feedforward, dropout) 
        self.encoder_frames = nn.TransformerEncoder(self.encoder_layer_frames, n_layers) 
        
        self.encoder_layer_bins = nn.TransformerEncoderLayer(frames, nhead,\
                                                        dim_feedforward, dropout)
        self.encoder_bins = nn.TransformerEncoder(self.encoder_layer_bins, n_layers)    
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor, input tensor, shape = [batch, frames, bins].

        Returns
        -------
        output_emb shape = [batch, frames', bins', out_chs].

        '''

        #batch, frames, bins = x.size
        x=x.permute(1, 0, 2)    # frames, batch, bins = x.size
        
        x = self.encoder_frames(x)
       
        x=x.permute(2, 1, 0)    # bins, batch, frames = x.size
        x = self.encoder_bins(x)
        
        x=x.permute(1, 2, 0)  # batch, frames, bins = x.size
        
        return x

class DPTrans(nn.Module):
    ''' Dual-Path Transformer'''
    def __init__(self, frames, bins,\
                 class_num, \
                 nhead, dim_feedforward, n_layers, dropout):
        '''
        Parameters
        ----------

        frames : int, the number of reduced frames.
        bins : int, input feature bins.
        
        class_num : int, number of classes.
        
        # d_model : int, the number of features in input.
        nhead : int, number of heads in multiheadattention.
        dim_feedforward : int, the dimension of feedforward network.
        n_layers : int, the number of encoder layers.
        dropout : float, dropout value.
                   
        frame_weight : float, weight of frame output.
        clip_weight : float, weight of clip output.
        
        reduce_type : str, optional 'max' or 'avg'.

        Returns
        -------
        None.

        '''
        super(DPTrans, self).__init__()
        
        # self.pe_layer = PositionalEncoding(bins, dropout)
        self.bn0 = nn.BatchNorm2d(bins)
        
        self.encoder1 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        self.encoder2 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        self.encoder3 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        # self.encoder4 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        
        self.spec_augmenter = SpecAugmentation(time_drop_width=4, time_stripes_num=2, 
                                               freq_drop_width=8, freq_stripes_num=2)
        
        # method 1
        self.linear = nn.Linear(bins, class_num)
        
        self.init_weight()
        
    def init_weight(self):
        
        init_bn(self.bn0) 
        init_layer(self.linear)
      
    def forward(self, x, spec_aug):
        '''
        Parameters
        ----------
        x : torch.Tensor, input tensor, shape = [batch, in_chs, frames, bins].

        Returns
        -------
        frame_output : torch.Tensor, output tensor, shape = [batch, frames, class_num].
        clip_output : torch.Tensor, output tensor, shape = [batch, class_num].
        weighted_clip_output : torch.Tensor, weighted output tensor, shape = [batch, class_num].

        '''  
        
        x = x.transpose(1, 3)   # x = [batch, bins, frames, in_chs]
        x = self.bn0(x)   # BN is done over the bins dimension
        x = x.transpose(1, 3)   # x = [batch, in_chs, frames, bins]
        
        if spec_aug:
            x = self.spec_augmenter(x)

        x = x.view(-1, x.size(2), x.size(3)) # x = [batch, frames, bins].
        
        x = self.encoder1(x) # x = [batch, frames, bins].
        x = self.encoder2(x) # x = [batch, frames, bins].
        x = self.encoder3(x) # x = [batch, frames, bins].
        # x = self.encoder4(x) # x = [batch, frames, bins].

        # x = torch.mean(x, dim=1)
        
        (output, _) = torch.max(x, dim=1)
        
        # x = x.reshape(x.size(0), -1)
        
        output = self.linear(output)   # frame_output = [batch, 100]

        return output

class SSDPT(nn.Module):
    ''' SSDPT'''
    def __init__(self, frames, bins,\
                 class_num, \
                 nhead, dim_feedforward, n_layers, dropout):
        '''
        Parameters
        ----------

        Returns
        -------
        None.

        '''
        super(SSDPT, self).__init__()
        
        # self.pe_layer = PositionalEncoding(bins, dropout)
        self.bn0 = nn.BatchNorm2d(bins)

        self.encoder1 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        self.encoder2 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        self.encoder3 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        # self.encoder4 = DPTrans_encoder(frames, bins, nhead, dim_feedforward, n_layers, dropout)
        
        self.spec_augmenter = SpecAugmentation(time_drop_width=4, time_stripes_num=4, 
                                               freq_drop_width=8, freq_stripes_num=2)
        
        
        
        # method 1
        self.linear = nn.Linear(bins, class_num)
        
        self.init_weight()
        
    def random_mask(self, x, mask_num, mask_size):
        B = x.shape[0]
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        
        for i in range(B):
            for j in range(mask_num):
                start_h = random.randrange(x.shape[1])
                start_w = random.randrange(x.shape[2])
                if start_h+mask_size<=x.shape[1]:
                    end_h = start_h+mask_size
                else:
                    end_h = x.shape[1]
                if start_w+mask_size<=x.shape[2]:
                    end_w = start_w+mask_size
                else:
                    end_w = x.shape[2] 
        
                mask_dense[i, start_h:end_h, start_w:end_w] = sys.float_info.epsilon
    
        x = x*mask_dense
        return x
        
    def init_weight(self):
        
        init_bn(self.bn0) 
        init_layer(self.linear)
      
    def forward(self, x, spec_aug=False):
        '''
        Parameters
        ----------
        x : torch.Tensor, input tensor, shape = [batch, in_chs, frames, bins].

        Returns
        -------

        '''  
        x = x.transpose(1, 3)   # x = [batch, bins, frames, in_chs]
        x = self.bn0(x)   # BN is done over the bins dimension
        x = x.transpose(1, 3)   # x = [batch, in_chs, frames, bins]
        
        x = x.view(-1, x.size(2), x.size(3))
        
        if spec_aug:
            # x = self.spec_augmenter(x)
            x = self.random_mask(x, mask_num=3, mask_size=5)
        
        # x = x.view(-1, x.size(2), x.size(3)) # x = [batch, frames, bins].
        aug_feat = x
        
        x = self.encoder1(x) # x = [batch, frames, bins].
        x = self.encoder2(x) # x = [batch, frames, bins].
        x = self.encoder3(x) # x = [batch, frames, bins].
        # x = self.encoder4(x) # x = [batch, frames, bins].

        # emb = torch.mean(x, dim=1)
        
        (emb, _) = torch.max(x, dim=1)
        
        # x = x.reshape(x.size(0), -1)
        
        output = self.linear(emb)   # frame_output = [batch, classes]

        return x, output
