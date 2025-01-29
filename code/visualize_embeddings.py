#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:50:17 2024

@author: itayta
"""


import torch
import esm
import huggingface_hub
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from esm.utils.constants import esm3 as C
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig





model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cpu") 

embs =  [x for x in model.encoder.sequence_embed.parameters()][0]
embmt = embs[0:len(C.SEQUENCE_VOCAB),:]
distmt = torch.cdist(embmt,embmt)


annotated_distmt = pd.DataFrame(distmt[4:24,4:24], 
                                columns=C.SEQUENCE_VOCAB[4:24], 
                                index=C.SEQUENCE_VOCAB[4:24])


tmp = annotated_distmt

for i in range(0,20):
    tmp.iloc[i,i] = np.max(annotated_distmt)

ax = sns.clustermap(tmp, cmap='viridis', xticklabels=1, yticklabels=1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.tick_params(axis='x', labelsize=6)  # x-axis tick labels
ax.tick_params(axis='y', labelsize=6)  #
plt.tight_layout()
plt.show()