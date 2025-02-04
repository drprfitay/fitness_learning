#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:37:28 2025

@author: itayta
"""

import torch
import os
import pandas as pd
from constants import *

from torch.utils.data import Dataset


class RawTokensDataset(Dataset):
    def __init__(self, dataset_name, mode=None, scale=True):
        
        self.sequence_df = pd.read_csv("%s/%s/sequences.csv" %  (DATASETS_PATH, dataset_name))
        self.unique_sequences = pd.unique(self.sequence_df["sequence"])
        self.dataset_name = dataset_name
        self.load = torch.load
        self.mode = mode
        self.scale = scale
                                         
    def __len__(self):
        return len(self.unique_sequences)

    def __getitem__(self, idx):
        seq = self.unique_sequences[idx]
        
        # Load data lazily (Example: Assume images in .png format)
        data = self.load("%s/%s/tokens/tokens_%s.pth" % (DATASETS_PATH, 
                                                         self.dataset_name, 
                                                         seq))
            
        if self.mode == "energy":
            dp = data["energy"].detach()
            if self.scale:
                #     dp = (dp - torch.min(dp, axis=0)[0]) / (torch.max(dp, axis=0)[0] - torch.min(dp, axis=0)[0])
                #dp = (dp - torch.mean(dp, axis=0)) / (torch.std(dp, axis=0))
                dp = dp[1:-2,:]
            return(dp)
        
        return data

