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



class EsmGfpDataset(Dataset):
    def __init__(self, 
                 sequences_path=None,
                 tokens_path=None,
                 embeddings_path=None,
                 embedding_layer=None,
                 mode=None):
        
        self.sequence_df = pd.read_csv(sequences_path)
        self.unique_sequences = pd.unique(self.sequence_df["sequence"])
        self.tokens_path = tokens_path
        self.embeddings_path = embeddings_path
        self.embedding_layer = embedding_layer
        self.load = torch.load
        self.mode = mode
        self.NO_ACTIVITY = 0
        self.LOW_GFP = 1
        self.HIGH_GFP = 2
        self.LOW_AMCYAN = 3
        self.HIGH_AMCYAN = 4
        self.LOW_ANY = 5
        self.HIGH_ANY = 6
        self.size = 4000
        self.sampled = torch.randperm(len(self.unique_sequences))[0:self.size]
                                         
    def __len__(self):
        #return self.size
        return len(self.unique_sequences)

    def __getitem__(self, idx):
        
        seq = self.unique_sequences[idx]        
        #seq = self.unique_sequences[self.sampled[idx]]
        data = self.load("%s/tokens_%s.pth" % (self.tokens_path, seq))
        
        #lbls = data["activity_labels_values"][0][2:10]
        #lbls[6] = 0 if lbls[6] + lbls[7] == 0 else 1        
        #lbls = lbls[:7]
        #lbls = lbls.tolist()
        
        lbls = data["activity_labels_values"][0][2:8].tolist()
        lbls = [int(data["activity_labels_values"][0][1])] + lbls
        # lbls =  torch.tensor(lbls, dtype=torch.float64)

        new_lbls = [0] * 7
        if sum(lbls) == 1: # if just 1 label on -> perfect 
            new_lbls = lbls
        elif lbls[self.HIGH_ANY] == 1: # if high_any -> set all but high any to zero
            new_lbls[self.HIGH_ANY] = 1 
        elif lbls[self.LOW_ANY] == 1: # high any is not set if this is true -> if low any set all but low to zero
            new_lbls[self.LOW_ANY] = 1 
        elif (lbls[self.HIGH_GFP] and lbls[self.HIGH_AMCYAN]) or\
             (lbls[self.HIGH_GFP] and lbls[self.LOW_AMCYAN]) or\
             (lbls[self.LOW_GFP] and lbls[self.HIGH_AMCYAN]):
            new_lbls[self.HIGH_ANY] =  1
        elif lbls[self.HIGH_GFP] and lbls[self.LOW_AMCYAN]:
            new_lbls[self.LOW_ANY] = 1 
        elif lbls[self.HIGH_GFP]:
            new_lbls[self.HIGH_GFP] = 1 
        elif lbls[self.LOW_GFP]:
            new_lbls[self.LOW_GFP] = 1 
        elif lbls[self.HIGH_AMCYAN]:
            new_lbls[self.HIGH_AMCYAN] = 1 
        elif lbls[self.LOW_AMCYAN]:
            new_lbls[self.LOW_AMCYAN] = 1          
            
        #new_lbls = [new_lbls[0]] + [sum(new_lbls[1:])]
        new_lbls = [new_lbls[0], sum(new_lbls[1:3]), sum(new_lbls[3:5]), sum(new_lbls[5:7])]
        lbls = torch.tensor(new_lbls, dtype=torch.float64)
            
        if self.mode == "embeddings":
            data = self.load("%s/embeddings_%s.pth" % (self.embeddings_path, seq))
            if self.embedding_layer is not None:
                return data["all_emb"][self.embedding_layer,:,:], lbls
            else:
                return (data["all_emb"], lbls)        
        
        
        if self.mode == "tokens":
            return(data)    
            
        if self.mode == "energy":
            dp = data["energy"].detach()
            dp = dp[1:-2,:]
            return(dp)
        
        return data
