import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init

import math
from PIL import Image
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc 

from lib.gcn_lib import Grapher as GCB 

class BSR(nn.Module):
    """
    A boundary-focused BSR module with:
      - multi-dilation (d=1,2,3)
      - difference of multi-dilated outputs
      - boundary attention gating
      - final fusion + residual (skip) connection
    """
    def __init__(self, in_channels, out_channels):
        super(BSR, self).__init__()

        # --- 1) Dilated Convolutions ---
        self.dilation1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=1, padding=1, bias=False)
        self.dilation2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=2, padding=2, bias=False)
        self.dilation3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   dilation=3, padding=3, bias=False)

        # --- 2) Convs to compute boundary attention from differences ---
        self.boundary_att_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),  # Single-channel attention
        )

        # --- 3) 1x1 conv to fuse boundary-dilations + original features ---
        fused_in_channels = (out_channels * 3) + (out_channels * 2)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(fused_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape (B, in_channels, H, W)
        Returns:
            Tensor of shape (B, out_channels, H, W) with boundary-focused features.
        """

        # --- (A) Multi-scale Dilation ---
        d1 = self.dilation1(x)  # (B, out_channels, H, W)
        d2 = self.dilation2(x)  # (B, out_channels, H, W)
        d3 = self.dilation3(x)  # (B, out_channels, H, W)

        # --- (B) Compute Differences to Highlight Boundaries ---
        diff_1_2 = abs(d2 - d1)
        diff_2_3 = abs(d3 - d2)

        # --- (C) Boundary Attention Map ---
        # Concatenate the difference maps and pass through a small conv to get a 1-channel attention
        boundary_diff_cat = torch.cat([diff_1_2, diff_2_3], dim=1)  # (B, out_channels*2, H, W)
        att_map = self.boundary_att_conv(boundary_diff_cat)         # (B, 1, H, W)
        att_map = torch.sigmoid(att_map)  # Range [0, 1], gating factor

        # --- (D) Gate the dilation outputs with the attention map ---
        # This step emphasizes boundary-like areas in each dilation output
        d1_att = d1 * att_map
        d2_att = d2 * att_map
        d3_att = d3 * att_map

        # --- (E) Final Fusion + Skip ---
        # We will fuse: 
        #   - Multi-dilated outputs (with attention) [d1_att, d2_att, d3_att]
        #   - Difference-based boundary signals [diff_1_2, diff_2_3]
        fused = torch.cat([d1_att, d2_att, d3_att, diff_1_2, diff_2_3], dim=1)

        output = self.final_fusion(fused) 
        return output

class UCB(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3, stride=1, padding=1, groups=1, activation='relu'):
        super(UCB,self).__init__()
        
        if(activation=='leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif(activation=='gelu'):
            self.activation = nn.GELU()
        elif(activation=='relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif(activation=='hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:    
            self.activation = nn.ReLU(inplace=True)
            
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_in,kernel_size=kernel_size,stride=stride,padding=padding,groups=groups,bias=True),
	    nn.BatchNorm2d(ch_in),
	    self.activation,
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0,bias=True),
           )

    def forward(self,x):
        x = self.up(x)
        return x    
    
class MSGBANET(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], drop_path_rate=0.0, img_size=224, k=11, padding=5, conv='mr', gcb_act='gelu', activation='relu'):
        super(MSGBANET, self).__init__()

        # Up-convolution block (UCB) parameters
        self.ucb_ks = 3
        self.ucb_pad = 1
        self.ucb_stride = 1
        self.activation = activation
        
        # Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} - default 'mr'
        self.gcb_act = gcb_act  # activation for GCB {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # normalization for GCB {batch, instance}
        self.bias = True  # bias for conv layers
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn
        self.epsilon = 0.2  # stochastic epsilon for GCB
        self.use_stochastic = False  # stochastic for GCB
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2]
        self.dpr = [self.drop_path] * 4  # stochastic depth decay rule
        self.num_knn = [self.k] * 4  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4


        # Graph Convolution Blocks (GCBs)
        self.gcb4 = nn.Sequential(GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                     self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4), drop_path=self.dpr[0],
                                     relative_pos=True, padding=self.padding))
        
        self.ucb3 = UCB(ch_in=channels[0], ch_out=channels[1], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[0], activation=self.activation)
        self.gcb3 = nn.Sequential(GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                     self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4), drop_path=self.dpr[1],
                                     relative_pos=True, padding=self.padding))
        
        self.ucb2 = UCB(ch_in=channels[1], ch_out=channels[2], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[1], activation=self.activation)
        self.gcb2 = nn.Sequential(GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                     self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // 4, drop_path=self.dpr[2],
                                     relative_pos=True, padding=self.padding))
        
        self.ucb1 = UCB(ch_in=channels[2], ch_out=channels[3], kernel_size=self.ucb_ks, stride=self.ucb_stride, padding=self.ucb_pad, groups=channels[2], activation=self.activation)
        self.gcb1 = nn.Sequential(GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act, self.gcb_norm,
                                     self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                                     relative_pos=True, padding=self.padding))
        
        # Boundary-Sensitive Refinement (BSR) Modules
        self.bsr4 = BSR(channels[0], channels[0])
        self.bsr3 = BSR(channels[1], channels[1])
        self.bsr2 = BSR(channels[2], channels[2])
        self.bsr1 = BSR(channels[3], channels[3])


    def forward(self, x, skips):      
        # GCAM4
        d4 = self.gcb4(x)
        d4 = self.bsr4(d4) * d4  # Apply BSR after d4

        
        # UCB3
        d3 = self.ucb3(d4)
        # Aggregation 3
        d3 = d3 + skips[0]
        d3 = self.gcb3(d3)
        d3 = self.bsr3(d3) * d3 # Apply BSR after d3

        
        # UCB2
        d2 = self.ucb2(d3)
        # Aggregation 2
        d2 = d2 + skips[1]
        d2 = self.gcb2(d2)
        d2 = self.bsr2(d2) * d2 # Apply BSR after d2

        
        # UCB1
        d1 = self.ucb1(d2)
        # Aggregation 1
        d1 = d1 + skips[2]
        d1 = self.gcb1(d1)
        d1 = self.bsr1(d1) * d1 # Apply BSR after d1

  
        return d4, d3, d2, d1

