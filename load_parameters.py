#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
def load_pars():
        parameter_dict={}

        ### train
        parameter_dict['reduce_type'] = 'max'
        parameter_dict['degrade_step'] = 1
        parameter_dict['load_ckpt'] = False
        parameter_dict['is_mixup'] = True
        parameter_dict['spec_aug'] = False
        parameter_dict['reshape'] = False
        parameter_dict['loss_weight'] = [0.001, 1]
        parameter_dict['score_weight'] = [0.001, 1]
        
        ### features                                                        
        # parameter_dict['feature_type'] = 'log_mel'                              # str, type of the acoustic features, choice = ['log_mel','STFT',HPSS']                     
        # parameter_dict['sample_rate'] = ''                                   # int, sample rate of loading audio, choice = [16000, 22050, 20480, 48000]     
        # parameter_dict['n_fft'] = 1024                                          # int, fft sample points, choice = [512, 1024, 2048] 
        # parameter_dict['hop_length'] = 512                                      # int, hop length of STFT, usually use window length//2 or window length//4
        # parameter_dict['win_length'] = 1024                                     # int, window length of STFT, usually use fft sample points
        # parameter_dict['bands_num'] = 64                                        # int, the number of filterbanks, choice = [40, 64, 128, 256] 
        # parameter_dict['fmin'] = 50                                             # int, the min frequency
        # parameter_dict['fmax'] = 11025                                          # int, the max frequency, usually use sample rate//2
        # parameter_dict['window'] = 'hann'                                       # str, window type of STFT, choice = ['triang', 'hamming', 'hann']            
        
        ### train
        parameter_dict['optimizer_eps'] = 1e-8
        parameter_dict['optimizer_betas'] = (0.9, 0.999)
        parameter_dict['weight_decay'] = 0.05
        
        parameter_dict['warmup_epochs'] = 2
        parameter_dict['scheduler_decay_epochs'] = 3
        parameter_dict['lr_scheduler_name'] = 'cosine'
        parameter_dict['warmup_lr'] = 5e-7
        
        ### DPT_encoder
        parameter_dict['nhead'] = 8
        parameter_dict['dim_feedforward'] = 32
        parameter_dict['n_layers'] = 1
        parameter_dict['dropout'] = 0

        return parameter_dict
    
