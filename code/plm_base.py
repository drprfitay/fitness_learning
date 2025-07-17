#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:22:42 2025

@author: itayta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:26:20 2025

@author: itayta
"""


import sys, os
import torch
import torch.nn.functional as F
import loralib as lora
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import einops
import yaml
import argparse

from esm_smart_dataset import *
from sequence_space_utils import *

from random import sample
from math import ceil
from collections import OrderedDict


global is_init
is_init = False
internal_wrapper = {"load_model": None}

class PlmWrapper():
    def unimplemented(self):
            raise NotImplementedError("Unimeplemted function")
            
    def __init__(self,
                 get_model_func=None,
                 get_tokenizer_func=None,
                 get_embeddings_func=None,
                 get_n_layers_func=None,
                 get_token_vocab_dim_func=None,
                 get_forward_func=None):
        
            self.get_model_func = get_model_func if get_model_func is not None else self.unimplemented
            self.get_embeddings_func = get_embeddings_func if get_embeddings_func is not None else self.unimplemented
            self.get_n_layers_func = get_n_layers_func if get_n_layers_func is not None else self.unimplemented
            self.get_tokenizer_func = get_tokenizer_func if get_tokenizer_func is not None else self.unimplemented
            self.get_token_vocab_dim_func = get_token_vocab_dim_func if get_token_vocab_dim_func is not None else self.unimplemented
            self.get_forward_func = get_forward_func if get_forward_func is not None else self.unimplemented
            
            

    def get_model(self):
        return self.get_model_func()
    
    def get_n_layers(self):
        return self.get_n_layers_func()
        
    def get_embeddings(self):
        return self.get_embeddings_func()
        
    def get_tokenizer(self):
        return self.get_tokenizer_func()
    
    def get_token_vocab_dim(self):
        return self.get_token_vocab_dim_func()
    
    def get_forward(self):
        return self.get_forward_func


def plm_init(PLM_BASE_PATH):
    #PLM_BASE_PATH = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
    ITAYFOLD_PATH = "%s/itayFold/" % PLM_BASE_PATH
    WEIGHTS_PATH = "/%s/weights/" % ITAYFOLD_PATH
        
    MODEL_WEIGHTS_FILE_NAME = "esm3/esm_model_weights.pth"
    LORA_WEIGHTS_FIlE_NAME =  "esm3/esm_lora_weights.pth"
    ENCODER_WEIGHTS_FILE_NAME = "esm3/structure_encoder.pth"
    DECODER_WEIGHTS_FILE_NAME = "esm3/structure_decoder.pth"

    
    def fix_esm_path():
        global original_sys_path
        
        # Specify the module name and path
        module_name = "esm"
        module_path = ITAYFOLD_PATH 
        
        # Store the original sys.path
        original_sys_path = sys.path.copy()
    
        # Temporarily add the local directory to sys.path
        sys.path.insert(0, os.path.abspath(module_path))
    
        # hack
        for mdl in [k for k,v in sys.modules.items() if module_name in k]:
            del sys.modules[mdl]
    
    fix_esm_path()
    
    import esm2
    import string
    
    global is_init
    is_init = True
    
    supported_esm2_models =\
            ["esm1_t34_670M_UR50S",
             "esm1_t34_670M_UR50D",
             "esm1_t34_670M_UR100",
             "esm1_t12_85M_UR50S",
             "esm1_t6_43M_UR50S",
             "esm1b_t33_650M_UR50S",
             "esm_msa1_t12_100M_UR50S",
             "esm_msa1b_t12_100M_UR50S",        
             "esm1v_t33_650M_UR90S_1",
             "esm1v_t33_650M_UR90S_2",
             "esm1v_t33_650M_UR90S_3",
             "esm1v_t33_650M_UR90S_4",
             "esm1v_t33_650M_UR90S_5",
             "esm_if1_gvp4_t16_142M_UR50",
             "esm2_t6_8M_UR50D",
             "esm2_t12_35M_UR50D",
             "esm2_t30_150M_UR50D",
             "esm2_t33_650M_UR50D",
             "esm2_t36_3B_UR50D",
             "esm2_t48_15B_UR50D"]
            
    def load_esm2_model_and_alphabet(model_name):            
        if model_name not in supported_esm2_models:
            raise BaseException("Unsupported model %s, model must be in: %s" %\
                                  (model_name, ", ".join(supported_esm2_models)))
            
        model_weights_and_data_path = "%s/esm2/%s.pth" % (WEIGHTS_PATH, model_name)
        
        if model_weights_and_data_path in os.listdir("%s/esm2" % WEIGHTS_PATH):
            model_data = torch.load(model_weights_and_data_path)
        else:    
            model_data, regression_data = esm2.pretrained._download_model_and_regression_data(model_name)
            
            if regression_data is not None:
                model_data["model"].update(regression_data["model"])
                
            # Save model data
            torch.save(model_data, model_weights_and_data_path)
            
            
        model, tokenizer =\
                esm2.pretrained.load_model_and_alphabet_core(model_name, 
                                                             model_data, 
                                                             regression_data=None)
                
        
        def get_esm_model():
            return model
        
        def get_esm_tokenizer():
            return tokenizer
        
        def get_embeddings():
            return model.embed_tokens
        
        def get_n_layers():
            return model.num_layers
            
        def get_token_vocab_dim():
            V, plm_d_model = model.embed_tokens.weight.size()
            all_toks = tokenizer.all_toks
            return all_toks, plm_d_model
        
        def forward_func(x):
            forward = model.forward(x, repr_layers=[model.last_layer])
            hh = forward["representations"][model.last_layer]
            logits = forward["logits"]
            return(logits, hh)
            
                    
                    
        return PlmWrapper(get_esm_model,
                          get_esm_tokenizer,
                          get_embeddings,
                          get_n_layers,        
                          get_token_vocab_dim,
                          forward_func)
    

    def load_model_internal(model_name):
        if model_name in supported_esm2_models:
            return load_esm2_model_and_alphabet(model_name)
        
    internal_wrapper["load_model"] = load_model_internal
        
def load_model(model_name): 
    if not is_init:
        raise BaseException("Please init PLM base first")
        
    return internal_wrapper["load_model"](model_name)
    
    
    
    