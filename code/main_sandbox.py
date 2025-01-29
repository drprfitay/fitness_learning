#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:43:09 2024

@author: itayta
"""




#

import huggingface_hub
import sys, os
import esm

# Specify the module name and path
module_name = "esm"
module_path = "/Users/itayta/Desktop/prot_stuff/itayFold"

# Store the original sys.path
original_sys_path = sys.path.copy()

# Temporarily add the local directory to sys.path
sys.path.insert(0, os.path.abspath(module_path))

# hack
for mdl in [k for k,v in sys.modules.items() if "esm" in k]:
    del sys.modules[mdl]

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig


#huggingface_hub.login(token="___")


model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cpu") 

prompt = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"
protein = ESMProtein(sequence=prompt)
# We can show the predicted structure for the generated sequence.
protein = model.generate(protein, GenerationConfig(track=["structure", "sequence"], num_steps=8))

a = 5


sys.path = original_sys_path