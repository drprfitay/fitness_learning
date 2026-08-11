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
import json
import torch
import torch.nn.functional as F

from utils import *

from random import sample
from math import ceil
from collections import OrderedDict
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, T5Tokenizer
from tokenizers import Tokenizer

global is_init

is_init = False
internal_wrapper = {"load_model": None}


def _resolve_lora_paths(lora_weights_path, lora_config_path=None):
    if lora_weights_path is None:
        return None, None

    lora_weights_path = os.path.abspath(str(lora_weights_path))
    if os.path.isdir(lora_weights_path):
        lora_config_path = lora_config_path or os.path.join(lora_weights_path, "training_config.json")
        candidate_weights = [
            os.path.join(lora_weights_path, "lora_weights.pt"),
            os.path.join(lora_weights_path, "final_model.pt"),
        ]
        for candidate in candidate_weights:
            if os.path.exists(candidate):
                return candidate, lora_config_path
        raise FileNotFoundError("could not find lora_weights.pt or final_model.pt under %s" % lora_weights_path)

    if lora_config_path is None:
        root, _ext = os.path.splitext(lora_weights_path)
        candidate = root + ".config.json"
        if os.path.exists(candidate):
            lora_config_path = candidate

    return lora_weights_path, lora_config_path


def _replace_linear_with_lora(module, r=8, alpha=16, dropout=0.0):
    try:
        import loralib as lora
    except ImportError as exc:
        raise ImportError("loralib is required to load LoRA weights") from exc

    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            replacement = lora.Linear(
                child.in_features,
                child.out_features,
                r=r,
                lora_alpha=alpha,
                lora_dropout=dropout,
                bias=child.bias is not None,
            ).to(device=child.weight.device, dtype=child.weight.dtype)
            replacement.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.data)
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            _replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout)


def apply_lora_weights_to_model(model,
                                lora_weights_path,
                                lora_config_path=None,
                                lora_r=None,
                                lora_alpha=None,
                                lora_dropout=None,
                                strict=False,
                                verbose=True):
    lora_weights_path, lora_config_path = _resolve_lora_paths(lora_weights_path, lora_config_path)
    if lora_weights_path is None:
        return model
    if not os.path.exists(lora_weights_path):
        raise FileNotFoundError("LoRA weights file does not exist: %s" % lora_weights_path)

    config = {}
    if lora_config_path is not None and os.path.exists(lora_config_path):
        with open(lora_config_path) as handle:
            config = json.load(handle)

    lora_r = int(lora_r if lora_r is not None else config.get("lora_r", 8))
    lora_alpha = int(lora_alpha if lora_alpha is not None else config.get("lora_alpha", 16))
    lora_dropout = float(lora_dropout if lora_dropout is not None else config.get("lora_dropout", 0.0))

    if verbose:
        print("[plm_base] applying LoRA: path=%s r=%d alpha=%d dropout=%.4g" %
              (lora_weights_path, lora_r, lora_alpha, lora_dropout))

    _replace_linear_with_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

    state = torch.load(lora_weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)

    loaded_lora_keys = sum(1 for key in state.keys() if "lora_" in key)
    unexpected_lora = [key for key in incompatible.unexpected_keys if "lora_" in key]
    if loaded_lora_keys == 0:
        raise ValueError("weights file does not contain LoRA tensors: %s" % lora_weights_path)
    if unexpected_lora or (strict and incompatible.unexpected_keys):
        raise RuntimeError("failed to load LoRA weights cleanly from %s" % lora_weights_path)

    if verbose:
        print("[plm_base] loaded LoRA tensors=%d missing=%d unexpected=%d" %
              (loaded_lora_keys, len(incompatible.missing_keys), len(incompatible.unexpected_keys)))

    return model

class PlmWrapper():
    def unimplemented(self):
            raise NotImplementedError("Unimeplemted function")
            
    def __init__(self,
                 get_model_func=None,
                 get_tokenizer_func=None,
                 get_embeddings_func=None,
                 get_n_layers_func=None,
                 get_token_vocab_dim_func=None,
                 encode_func=None,
                 forward_func=None):
        
            self.get_model_func = get_model_func if get_model_func is not None else self.unimplemented
            self.get_embeddings_func = get_embeddings_func if get_embeddings_func is not None else self.unimplemented
            self.get_n_layers_func = get_n_layers_func if get_n_layers_func is not None else self.unimplemented
            self.get_tokenizer_func = get_tokenizer_func if get_tokenizer_func is not None else self.unimplemented
            self.get_token_vocab_dim_func = get_token_vocab_dim_func if get_token_vocab_dim_func is not None else self.unimplemented
            self.encode_func = encode_func if encode_func is not None else self.unimplemented
            self.forward_func = forward_func if forward_func is not None else self.unimplemented
            
            

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

    def get_encode(self):
        return self.encode_func
    
    def get_forward(self):
        return self.forward_func

    def load_lora_weights(self, *args, **kwargs):
        apply_lora_weights_to_model(self.get_model(), *args, **kwargs)
        return self


def plm_init(PLM_BASE_PATH):
    #PLM_BASE_PATH = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
    MODELS_PATH = "%s/models/" % PLM_BASE_PATH
    WEIGHTS_PATH = "/%s/weights/" % MODELS_PATH
        
    MODEL_WEIGHTS_FILE_NAME = "esm3/esm_model_weights.pth"
    LORA_WEIGHTS_FIlE_NAME =  "esm3/esm_lora_weights.pth"
    ENCODER_WEIGHTS_FILE_NAME = "esm3/structure_encoder.pth"
    DECODER_WEIGHTS_FILE_NAME = "esm3/structure_decoder.pth"

    
    def fix_esm_path():
        global original_sys_path
        
        # Specify the module name and path
        module_name = "esm"
        module_path = MODELS_PATH 
        
        # Store the original sys.path
        original_sys_path = sys.path.copy()
    
        # Temporarily add the local directory to sys.path
        sys.path.insert(0, os.path.abspath(module_path))
    
        # hack
        for mdl in [k for k,v in sys.modules.items() if module_name in k]:
            del sys.modules[mdl]
    
    fix_esm_path()
    
    import esm2
    from progen.modeling_progen import ProGenForCausalLM
    from progen.configuration_progen import ProGenConfig
    
    global is_init
    is_init = True
    
    supported_ablang_models = ["igbert"]
    supported_esm2_models = ["esm1_t34_670M_UR50S", "esm1_t34_670M_UR50D", "esm1_t34_670M_UR100",
                             "esm1_t12_85M_UR50S", "esm1_t6_43M_UR50S", "esm1b_t33_650M_UR50S",
                             "esm_msa1_t12_100M_UR50S","esm_msa1b_t12_100M_UR50S","esm1v_t33_650M_UR90S_1",
                             "esm1v_t33_650M_UR90S_2", "esm1v_t33_650M_UR90S_3", "esm1v_t33_650M_UR90S_4",
                             "esm1v_t33_650M_UR90S_5", "esm_if1_gvp4_t16_142M_UR50","esm2_t6_8M_UR50D",
                             "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D", "esm2_t33_650M_UR50D",
                             "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]
    supported_progen_models = ["progen2-small", "progen2-medium"]
    supported_transformers_pretrained_models = ["prot_bert", "ankh3-large"]
    supported_saprot_models = [
        "saprot",
        "saprot_35m_af2",
        "saprot_650m_af2",
        "saprot_650m_pdb",
        "saprot_1.3b_afdb_omg_ncbi",
    ]

    def load_ablang_model_and_alphabet(model_name):
        if model_name not in supported_ablang_models:
            raise BaseException("Unsupported model %s, model must be in: %s" %\
                                  (model_name, ", ".join(supported_ablang_models)))


        abland_transformers_kwargs_dictionary = {
            "igbert": {
                "tokenizer_kwargs": {
                    "do_lower_case": False,
                },
                "model_kwargs": {
                    "add_pooling_layer": False
                },
                "name": "Exscientia/IgBert"
                }
        }
        
        model = BertModel.from_pretrained(abland_transformers_kwargs_dictionary[model_name]["name"], 
                                         **abland_transformers_kwargs_dictionary[model_name]["model_kwargs"])
                                         
        tokenizer = BertTokenizer.from_pretrained(abland_transformers_kwargs_dictionary[model_name]["name"],
                                                  **abland_transformers_kwargs_dictionary[model_name]["tokenizer_kwargs"])

        def get_ablang_model():
            return model
        
        def get_ablang_tokenizer():
            return tokenizer
        
        def get_embeddings():
            return model.embeddings
        
        def get_n_layers():
            return len(model.encoder.layer)
            
        def get_token_vocab_dim():
            V, abland_d_model = model.embeddings.word_embeddings.weight.size()
            all_toks = tokenizer.vocab
            return all_toks, abland_d_model

        def encode_func(seq):
            seq = " ".join([aa for aa in seq])
            final_seq = seq.replace("-", "[PAD]")
            return tokenizer.encode(final_seq)
        
        def forward_func(x, attention_mask=None):
            # You can add an attention mask, but in our case we don't need it
            if attention_mask is not None:
                forward = model.forward(x, attention_mask=attention_mask)
            else:
                forward = model.forward(x)
            hh = forward.last_hidden_state
            logits = None # TODO: add logits
            return(logits, hh)                                
                    
        return PlmWrapper(get_ablang_model,
                          get_ablang_tokenizer,
                          get_embeddings,
                          get_n_layers,        
                          get_token_vocab_dim,
                          encode_func,
                          forward_func)
               
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

        def encode_func(seq):
            seq = "<cls>" + seq + "<eos>"
            return tokenizer.encode(seq)
        
        def forward_func(x):
            forward = model.forward(x, repr_layers=[model.num_layers])
            hh = forward["representations"][model.num_layers]
            logits = forward["logits"]
            return(logits, hh)                                
                    
        return PlmWrapper(get_esm_model,
                          get_esm_tokenizer,
                          get_embeddings,
                          get_n_layers,        
                          get_token_vocab_dim,
                          encode_func,
                          forward_func)

    def load_progen_model_and_alphabet(model_name):   

        if model_name not in supported_progen_models:
            raise BaseException("Unsupported model %s, model must be in: %s" %\
                                  (model_name, ", ".join(supported_progen_models)))
            
        weights_path = "%s/progen/%s_weights.pth" % (WEIGHTS_PATH, model_name)
        config_path = "%s/progen/%s_config.json" % (WEIGHTS_PATH, model_name)
        tokenizer_config_path = "%s/progen/%s_tokenizer.json" % (WEIGHTS_PATH, model_name)
        
        config = ProGenConfig.from_pretrained(config_path)
        model = ProGenForCausalLM(config)
        model.load_state_dict(torch.load(weights_path))
        model.eval()
        
        with open(tokenizer_config_path, 'r') as f:
            tokenizer = Tokenizer.from_str(f.read())
                    
        def get_progen_model():
            return model
        
        def get_progen_tokenizer():
            return tokenizer
        
        def get_embeddings():
            return model.transformer.wte
        
        def get_n_layers():
            return config.n_layer
            
        def get_token_vocab_dim():            
            return tokenizer.get_vocab(), config.n_embd
           
        def encode_func(seq):         
            final_seq = "<|bos|>" + seq + "<|eos|>"   
            return tokenizer.encode(final_seq).ids
        
        def forward_func(x):
            forward = model(x, output_hidden_states=True)
            hh = forward.hidden_states[-1]
            logits = forward.logits
            return(logits, hh)                                
                    
        return PlmWrapper(get_progen_model,
                          get_progen_tokenizer,
                          get_embeddings,
                          get_n_layers,        
                          get_token_vocab_dim,
                          encode_func,
                          forward_func)
    
    def load_saprot_model_and_alphabet(model_name):
        if model_name not in supported_saprot_models:
            raise BaseException("Unsupported model %s, model must be in: %s" %\
                                (model_name, ", ".join(supported_saprot_models)))

        full_name_dict = {
            "saprot": "westlake-repl/SaProt_650M_AF2",
            "saprot_35m_af2": "westlake-repl/SaProt_35M_AF2",
            "saprot_650m_af2": "westlake-repl/SaProt_650M_AF2",
            "saprot_650m_pdb": "westlake-repl/SaProt_650M_PDB",
            "saprot_1.3b_afdb_omg_ncbi": "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
        }

        model_path = full_name_dict[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=False)
        model = AutoModelForMaskedLM.from_pretrained(model_path, token=False)

        def get_saprot_model():
            return model

        def get_saprot_tokenizer():
            return tokenizer

        def get_embeddings():
            return model.get_input_embeddings()

        def get_n_layers():
            return model.config.num_hidden_layers

        def get_token_vocab_dim():
            return tokenizer.get_vocab(), model.config.hidden_size

        def encode_func(seq):
            return tokenizer(seq)["input_ids"]

        def forward_func(x, attention_mask=None):
            if x.dim() == 1:
                x = x.unsqueeze(0)

            if attention_mask is None:
                attention_mask = torch.ones_like(x)

            forward = model(
                input_ids=x,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return forward.logits, forward.hidden_states[-1]

        return PlmWrapper(get_saprot_model,
                          get_saprot_tokenizer,
                          get_embeddings,
                          get_n_layers,
                          get_token_vocab_dim,
                          encode_func,
                          forward_func)

    # TODO: merge with ablang
    def load_transformers_pretrained_model_and_alphabet(model_name):

        if model_name not in supported_transformers_pretrained_models:
            raise BaseException("Unsupported model %s, model must be in: %s" %\
                                  (model_name, ", ".join(supported_transformers_pretrained_models)))    

        full_name_dict = {"prot_bert": "Rostlab/prot_bert", "ankh3-large": "ElnaggarLab/ankh3-large"}    
        
        model = AutoModel.from_pretrained(full_name_dict[model_name], token=False)

        def get_encoder(model_name, model):
            if model_name == "prot_bert":
                return (model, 
                        model.embeddings.word_embeddings, 
                        model.encoder.layer, 
                        True,
                        AutoTokenizer.from_pretrained(full_name_dict[model_name], token=False))
            elif model_name == "ankh3-large":
                return (model.encoder, 
                        model.encoder.embed_tokens, 
                        model.encoder.block, 
                        False,
                        T5Tokenizer.from_pretrained(full_name_dict[model_name], token=False, legacy=False))
            else:
                raise BaseException("Unsupported model %s, model must be in: %s" %\
                                  (model_name, ", ".join(supported_transformers_pretrained_models)))
        
        encoder, embeddings, layers, add_space_to_seq, tokenizer = get_encoder(model_name, model)
        N_layers = len(layers)

        def get_transformer_encoder_model():
            return encoder
        
        def get_transformer_encoder_tokenizer():
            return tokenizer
        
        def get_embeddings():
            return embeddings
        
        def get_n_layers():
            return N_layers
            
        def get_token_vocab_dim():            
            return tokenizer.get_vocab(), embeddings.weight.shape[1]
           
        def encode_func(seq): 
            if add_space_to_seq:
                seq = " ".join([aa for aa in seq])     

            return tokenizer(seq)["input_ids"]
        
        def forward_func(x):
            if x.dim() == 1:
                x = x.unsqueeze(0)

            attention_mask = torch.ones(x.shape).to(x.device)
            forward = encoder(input_ids=x, attention_mask=attention_mask)
            hh = forward.last_hidden_state

            return(None, hh)                                
                    
        return PlmWrapper(get_transformer_encoder_model,
                          get_transformer_encoder_tokenizer,
                          get_embeddings,
                          get_n_layers,        
                          get_token_vocab_dim,
                          encode_func,
                          forward_func)
            

    def load_model_internal(model_name):
        if model_name in supported_esm2_models:
            return load_esm2_model_and_alphabet(model_name)

        if model_name in supported_ablang_models:
            return load_ablang_model_and_alphabet(model_name)

        if model_name in supported_progen_models:
            return load_progen_model_and_alphabet(model_name)

        if model_name in supported_transformers_pretrained_models:
            return load_transformers_pretrained_model_and_alphabet(model_name)

        if model_name in supported_saprot_models:
            return load_saprot_model_and_alphabet(model_name)
        
    internal_wrapper["load_model"] = load_model_internal
        
def load_model(model_name,
               lora_weights_path=None,
               lora_config_path=None,
               lora_r=None,
               lora_alpha=None,
               lora_dropout=None,
               lora_strict=False,
               lora_verbose=True): 
    if not is_init:
        raise BaseException("Please init PLM base first")

    plm_obj = internal_wrapper["load_model"](model_name)
    if lora_weights_path is not None:
        apply_lora_weights_to_model(
            plm_obj.get_model(),
            lora_weights_path=lora_weights_path,
            lora_config_path=lora_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            strict=lora_strict,
            verbose=lora_verbose,
        )

    return plm_obj
      
# This is kinda deprecated at this point, better using plmEmbeddingModel instead
class plmTrunkModel(torch.nn.Module):    
    def __init__(self, 
                 plm_name,
                 hidden_layers=[1024],
                 activation="relu",
                 opmode="pos",
                 emb_only=True,
                 logits_only=False,
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 tok_dropout=True,
                 specific_pos=None,
                 kernel_size=20,
                 stride=5,
                 trunk_classes=2,
                 device=torch.device("cpu"),                 
                 dtype=torch.double,
                 lora_weights_path=None,
                 lora_config_path=None,
                 lora_r=None,
                 lora_alpha=None,
                 lora_dropout=None,
                 lora_strict=False,
                 lora_verbose=True):
        super().__init__()
        
        
        # plm = load_model(plm_name)
        # #plm, plm_tokenizer = load_esm2_model_and_alphabet(plm_name)
        # V, plm_d_model = plm.embed_tokens.weight.size()
        
        plm_obj = load_model(
            plm_name,
            lora_weights_path=lora_weights_path,
            lora_config_path=lora_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_strict=lora_strict,
            lora_verbose=lora_verbose,
        )
        plm = plm_obj.get_model()
        plm_tokenizer = plm_obj.get_tokenizer()
        vocab, plm_d_model = plm_obj.get_token_vocab_dim()
        V = len(vocab)
        
        self.tokenizer = plm_tokenizer
        self.plm = plm.to(device)
        self.last_layer = plm_obj.get_n_layers()
        self.forward_func = plm_obj.get_forward()
        self.internal_encode = plm_obj.get_encode()
        self.specific_pos = specific_pos
        self.vocab = vocab
        
        #### IMPORTANT -> this is a bunch of legacy code that I got too scared to delete - I will delete it soon

        # if (type(plm) == esm2.model.esm2.ESM2):
        #     self.last_layer = plm_obj.get_n_layers()
            
        # def plm_forward_presentation(x):
        #     forward = self.plm.forward(x, repr_layers=[self.last_layer])
        #     hh = forward["representations"][self.last_layer]
        #     return(hh)
            
        # self.forward_func = plm_forward_presentation                
        # self.opmode = opmode
        
        # possible_opmodes = ["mean", "class", "avgpool", "pos"]
        
        # if opmode not in possible_opmodes:
        #     raise Exception("Unable to support opmode %s for trunk model, allowed opmodes are: %s" % (opmode, ", ".join(possible_opmodes)))
                        
        # if opmode == "mean":            
        #     if specific_pos is not None:
        #         # Average across specific positions
        #         self.specific_pos = torch.tensor(specific_pos, dtype=torch.int64) - 1 # PDB INDEX!!!!!! (1-based)
                
        #         def emb_pool_func(hh):                
        #             return(hh[:,self.specific_pos,:].mean(dim=1))
        #     else:
        #         def emb_pool_func(hh):                
        #             return(hh.mean(dim=1))
            
        # elif opmode == "class":
        #     class_token = torch.tensor(self.tokenizer.encode("<unk>"), dtype=torch.int64)
            
        #     def emb_pool_func(hh):
        #         return(hh[:,0,:])
            
        # elif opmode == "avgpool":
        #     self.conv1d = torch.nn.AvgPool1d(kernel_size=kernel_size,stride=stride)
                
        #     def emb_pool_func(hh):
        #         return(self.conv1d(einops.rearrange(hh,"B S D->B D S")).mean(dim=2))   
        
        # elif opmode == "pos":
        #     self.specific_pos = torch.tensor(specific_pos, dtype=torch.int64) - 1 # PDB INDEX!!!!!! (1-based)
            
        #     def emb_pool_func(hh):
        #         return(hh[:,self.specific_pos,:].flatten(1,2))
            
            
        # trunk_d_in_factor = 1 if opmode != "pos" else len(self.specific_pos)
        trunk_d_in_factor = 1
            
            
        # self.emb_func = emb_pool_func
        self.epinnet_trunk = EpiNNet(d_in=plm_d_model * trunk_d_in_factor,
                                     d_out=trunk_classes,                 
                                     hidden_layers=hidden_layers,
                                     activation=activation,
                                     layer_norm=layer_norm,
                                     use_bias=use_bias,
                                     activation_on_last_layer=activation_on_last_layer,
                                     device=device,                 
                                     dtype=dtype).to(device)

        if emb_only:
            self.final_forward = self._emb_only_forward            
        elif logits_only:
            self.final_forward = self._logits_only_forward 
        else:   
            self.final_forward = self._forward
        
    def encode(self, seq):            
        enc_seq = ""
        if self.opmode == "class":
            enc_seq = "<unk>"
                
        enc_seq = enc_seq + "<cls>" + seq + "<eos>"
                
        return self.tokenizer.encode(enc_seq)
            
    def _logits_only_forward(self, x):
        return self.forward_func(x)[0]

    def _emb_only_forward(self, x):
        return self.forward_func(x)[1]

    def _forward(self, x):                
        hh = self._emb_only_forward(x)

        emb = torch.nn.functional.normalize(hh[:,torch.tensor(self.specific_pos),:], dim=1).mean(dim=1)
        emb = torch.nn.functional.normalize(emb, dim=1)
            
        return emb, hh, self.epinnet_trunk(emb)
    
    def forward(self, x):
        return self.final_forward(x)
      
class plmEmbeddingModel(torch.nn.Module):
    def __init__(self, 
                 plm_name,
                 emb_only=True,
                 logits_only=False,
                 tok_dropout=True,
                 device=torch.device("cpu"),                 
                 dtype=torch.double,
                 lora_weights_path=None,
                 lora_config_path=None,
                 lora_r=None,
                 lora_alpha=None,
                 lora_dropout=None,
                 lora_strict=False,
                 lora_verbose=True):
        super().__init__()
        plm_obj = load_model(
            plm_name,
            lora_weights_path=lora_weights_path,
            lora_config_path=lora_config_path,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_strict=lora_strict,
            lora_verbose=lora_verbose,
        )
        plm = plm_obj.get_model()
        plm_tokenizer = plm_obj.get_tokenizer()
        vocab, plm_d_model = plm_obj.get_token_vocab_dim()
        V = len(vocab)

        self.plm_name = plm_name
        self.tokenizer = plm_tokenizer
        self.plm = plm.to(device)
        self.last_layer = plm_obj.get_n_layers()
        self.forward_func = plm_obj.get_forward()
        self.internal_encode = plm_obj.get_encode()
        self.vocab = vocab
        
        if emb_only:
            self.final_forward = self._emb_only_forward            
        elif logits_only:
            self.final_forward = self._logits_only_forward

    def encode(self, seq):
        # enc_seq = ""
        # enc_seq = enc_seq + "<cls>" + seq + "<eos>"
        # return self.tokenizer.encode(enc_seq)
        return self.internal_encode(seq)



    def _logits_only_forward(self, x, **kwargs):
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            return self.forward_func(x, attention_mask=attention_mask)[0]
        else:
            return self.forward_func(x)[0]

    def _emb_only_forward(self, x, **kwargs):
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            return self.forward_func(x, attention_mask=attention_mask)[1]
        else:
            return self.forward_func(x)[1]
    
    def forward(self, x, **kwargs):
        return self.final_forward(x, **kwargs)

class StructurePlmEmbedding(plmEmbeddingModel):
    """PLM wrapper using structure tokens supplied in PDB-sequence coordinates."""

    def __init__(self,
                 plm_name,
                 wt_sequence,
                 pdb_sequence,
                 foldseek_tokens,
                 **kwargs):
        super().__init__(plm_name, **kwargs)

        self.wt_sequence = wt_sequence.upper()
        self.pdb_sequence = pdb_sequence.upper()
        self.foldseek_tokens = foldseek_tokens.lower()

        if len(self.pdb_sequence) != len(self.foldseek_tokens):
            raise ValueError(
                "PDB sequence and PDB token lengths differ: %d != %d" %
                (len(self.pdb_sequence), len(self.foldseek_tokens))
            )

        self.structure_sequence, self.structure_mapping = self._align_structure_to_wt()

        mapped = [i for i in self.structure_mapping if i is not None]
        self.structure_coverage = len(mapped) / len(self.wt_sequence) if self.wt_sequence else 0.0
        self.alignment_identity = (
            sum(self.wt_sequence[wt_i] == self.pdb_sequence[pdb_i]
                for wt_i, pdb_i in enumerate(self.structure_mapping)
                if pdb_i is not None) / len(mapped)
            if mapped else 0.0
        )

    def _align_structure_to_wt(self):
        """Semi-global alignment covering every WT position.

        Extra PDB residues at either terminus are free and ignored. Internal PDB
        insertions are skipped; WT residues absent from the PDB receive '#'.
        """
        wt = self.wt_sequence
        pdb = self.pdb_sequence
        n, m = len(wt), len(pdb)

        match_score = 2
        mismatch_score = -1
        gap_score = -2

        dp = [[0] * (m + 1) for _ in range(n + 1)]
        trace = [[None] * (m + 1) for _ in range(n + 1)]

        # PDB prefix is free; every WT position still has to be represented.
        for j in range(1, m + 1):
            trace[0][j] = "pdb"
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + gap_score
            trace[i][0] = "wt"

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diagonal = dp[i - 1][j - 1] + (
                    match_score if wt[i - 1] == pdb[j - 1] else mismatch_score
                )
                wt_gap = dp[i - 1][j] + gap_score       # WT residue missing in PDB
                pdb_gap = dp[i][j - 1] + gap_score      # extra PDB residue

                # Prefer a residue-residue alignment on ties.
                best = max(diagonal, wt_gap, pdb_gap)
                dp[i][j] = best
                if diagonal == best:
                    trace[i][j] = "diag"
                elif wt_gap == best:
                    trace[i][j] = "wt"
                else:
                    trace[i][j] = "pdb"

        # PDB suffix is free.
        j = max(range(m + 1), key=lambda x: dp[n][x])
        i = n
        mapping = [None] * n

        while i > 0:
            move = trace[i][j]
            if move == "diag":
                mapping[i - 1] = j - 1
                i -= 1
                j -= 1
            elif move == "wt":
                # WT residue has no structural residue.
                i -= 1
            elif move == "pdb":
                # PDB insertion relative to WT: ignore its structural token.
                j -= 1
            else:
                raise RuntimeError("Failed to trace WT/PDB alignment")

        structure_sequence = "".join(
            self.foldseek_tokens[pdb_i] if pdb_i is not None else "#"
            for pdb_i in mapping
        )
        return structure_sequence, mapping

    def encode(self, seq=None):
        if seq is None:
            seq = self.wt_sequence
        seq = seq.upper()

        if len(seq) != len(self.wt_sequence):
            raise ValueError(
                "Sequence and WT lengths differ: %d != %d" %
                (len(seq), len(self.wt_sequence))
            )

        combined_seq = "".join(
            aa + structure_token
            for aa, structure_token in zip(seq, self.structure_sequence)
        )
        return self.internal_encode(combined_seq)

class abPlmEmbeddingModel(plmEmbeddingModel):
    def esm_encode(self, seq):
        h_l_split = seq.split("#L#")
        h_seq = h_l_split[0]
        l_seq = h_l_split[1]
        h_seq = h_seq.split("#H#")[1]
        final_seq = h_seq + l_seq
        final_seq = final_seq.replace("-", "<pad>")
        return self.tokenizer.encode(final_seq)

    def encode_heavy_or_light_only(self, seq):
        seq = " ".join([aa for aa in seq])
        final_seq = seq.replace("-", "[PAD]")
        return self.tokenizer.encode(final_seq)

    def encode(self, seq):
        h_l_split = seq.split("#L#")
        h_seq = h_l_split[0]
        l_seq = h_l_split[1]
        h_seq = h_seq.split("#H#")[1]
        h_seq = " ".join([aa for aa in h_seq])
        l_seq = " ".join([aa for aa in l_seq])
        final_seq = (h_seq + " [SEP] " + l_seq).replace("-", "[PAD]")
        return self.tokenizer.encode(final_seq)

class EpiNNet(torch.nn.Module):
    def __init__(self, 
                 d_in,
                 d_out,                 
                 hidden_layers=[1024],
                 activation="sigmoid",
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 device=torch.device("cpu"),                 
                 dtype=torch.double,
                 **kwargs):
        super().__init__()
        
        sequence_list = []
        
        activation_dict = {'relu': torch.nn.ReLU(),
                           'gelu': torch.nn.GELU(),
                           'sigmoid': torch.nn.Sigmoid()}
        
        if activation not in activation_dict.keys():
            activation = 'sigmoid'
            
        activation_func = activation_dict[activation]
        
        layers = [d_in] + hidden_layers + [d_out]
        
        N_layers = len(layers) - 1
        for layer_idx in range(0, N_layers):                        
            l_in = layers[layer_idx]
            l_out = layers[layer_idx + 1]
            
            if layer_norm:
                sequence_list += [('l%d_norm' % layer_idx, torch.nn.LayerNorm(l_in))]
            
            sequence_list += [('l%d_linear' % layer_idx, torch.nn.Linear(l_in, l_out, use_bias))]
            
            # last layer
            if layer_idx != (N_layers - 1) or activation_on_last_layer:            
                    sequence_list += [('l%d_activation' % layer_idx, activation_func)]
            
            
        self.sequential = torch.nn.Sequential(OrderedDict(sequence_list)).to(device)
    
    def forward(self, x):
        return self.sequential(x)
            
class seqMLP(torch.nn.Module):
    def __init__(self, 
                 encoding_type,
                 encoding_size,
                 encoding_func,
                 plm_name=None,                         
                 hidden_layers=[1024],
                 activation="sigmoid",
                 opmode="pos",
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 tok_dropout=True,
                 device=torch.device("cpu"),                 
                 dtype=torch.double):
        super().__init__()
                 
        possible_encodings = ["onehot", "plm_embedding"]
         
        if encoding_type not in possible_encodings:
            raise Exception("Unable to support encoding type %s for trunk model, allowed encoding types are: %s" % (encoding_type, ", ".join(possible_encodings)))
        
        self.encoding_type = encoding_type
        self.encoding_size = encoding_size
        
        
        if encoding_type == "plm_embedding":
            plm_obj = load_model(plm_name)
            vocab, plm_d_model = plm_obj.get_token_vocab_dim()
            V = len(vocab)
            #plm, plm_tokenizer = load_esm2_model_and_alphabet(plm_name)
            #V, plm_d_model = plm.embed_tokens.weight.size()
                    
            self.tokenizer = plm_obj.get_tokenizer()
            self.encoding_func = encoding_func # Should return just requested positiosn working on
            
            def encode(seq):                
                selected_seq = self.encoding_func(seq)
                return self.tokenizer.encode("".join(selected_seq))
            
            
            self.embedding = torch.nn.Embedding(V, plm_d_model)
            
            def forward(self, x):                
                return self.epinnet_trunk(x)
            
            d_in = plm_d_model * self.encoding_size  # should be num of working positions * d_model
           
        elif encoding_type == "onehot":
            self.encoding_func = encoding_func # Should return one hot encoding
            
            def encode(self, seq):
                return self.encoding_fun(seq)
            
            
            def forward(x):                
                return self.epinnet_trunk(x)
            
            d_in = self.encoding_size # Should be overall dimension of onehot
            
        self.encode_int = encode
        self.epinnet_trunk = EpiNNet(d_in=d_in,
                                     d_out=1,                 
                                     hidden_layers=hidden_layers,
                                     activation=activation,
                                     layer_norm=layer_norm,
                                     use_bias=use_bias,
                                     activation_on_last_layer=activation_on_last_layer,
                                     device=device,                 
                                     dtype=dtype).to(device)

    def encode(self, *args):
        return self.encode_int(*args)
