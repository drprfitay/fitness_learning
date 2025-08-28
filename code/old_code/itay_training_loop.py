#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 15:43:18 2024

@author: itayta
"""

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
import random
import torch
import torch.nn.functional as F
import loralib as lora
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse





parser = argparse.ArgumentParser(description="ItayFold main")


parser.add_argument("--is_sequence_indices", type=bool, help="Flag to determine whether indices are numerical or full sequences (pseudo sequeces)", required=False, default=False)
parser.add_argument("--mode", type=str, help="'scheduler' or 'worker'", required=True)
parser.add_argument("--indices", nargs='+', help='List of indices to work on')
parser.add_argument("--index_range", nargs='+', help='Range of indices to work on', default="")
parser.add_argument("--operations", nargs='+', help='List of indices to work on', default=[])




# Parse the arguments
#args = parser.parse_args()



from constants import *
from fix_esm_path import fix_esm_path

fix_esm_path()


from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.tokenization import get_model_tokenizers



from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
)

from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)

#from esm.tokenization import get_esm3_model_tokenizers
from esm.tokenization.function_tokenizer import (
    InterProQuantizedTokenizer as EsmFunctionTokenizer,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)

from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.types import FunctionAnnotation


huggingface_hub.login(token=HF_TOKEN)


def assign_lora_weights(model, 
                        save_lora_weights=False,
                        print_parameters=True,
                        train_bias=False):
    
    if LORA_TRANSFORMER_LINEAR_WEIGHTS:
        
        for transformer_block_idx in range(0, N_LAYERS):            
            model.transformer.blocks[transformer_block_idx].attn.layernorm_qkv[1] =\
                lora.Linear(model.transformer.blocks[transformer_block_idx].attn.layernorm_qkv[1].in_features, 
                            model.transformer.blocks[transformer_block_idx].attn.layernorm_qkv[1].out_features, 
                            LORA_R)    
            model.transformer.blocks[transformer_block_idx].attn.out_proj =\
                lora.Linear(model.transformer.blocks[transformer_block_idx].attn.out_proj.in_features, 
                            model.transformer.blocks[transformer_block_idx].attn.out_proj.out_features, 
                            LORA_R)    
            model.transformer.blocks[transformer_block_idx].ffn[1] =\
                lora.Linear(model.transformer.blocks[transformer_block_idx].ffn[1].in_features, 
                            model.transformer.blocks[transformer_block_idx].ffn[1].out_features, 
                            LORA_R)    
            model.transformer.blocks[transformer_block_idx].ffn[3] =\
                lora.Linear(model.transformer.blocks[transformer_block_idx].ffn[3].in_features, 
                            model.transformer.blocks[transformer_block_idx].ffn[3].out_features, 
                            LORA_R)                
            
            if transformer_block_idx == 0:
                model.transformer.blocks[transformer_block_idx].geom_attn.proj =\
                    lora.Linear(model.transformer.blocks[transformer_block_idx].geom_attn.proj.in_features, 
                                model.transformer.blocks[transformer_block_idx].geom_attn.proj.out_features, 
                                LORA_R)    
                
                model.transformer.blocks[transformer_block_idx].geom_attn.out_proj =\
                    lora.Linear(model.transformer.blocks[transformer_block_idx].geom_attn.out_proj.in_features, 
                                model.transformer.blocks[transformer_block_idx].geom_attn.out_proj.out_features, 
                                LORA_R)    


    if train_bias:
        bias_str = "all"
    else:
        bias_str = "none"
        
    lora.mark_only_lora_as_trainable(model,
                                     bias=bias_str)
    
    if print_parameters:
        
        params  = [p.numel() for p in model.parameters()]
        trainable_params = [p.numel() for p in model.parameters() if p.requires_grad==True]
        
        
        print("Training %d of %d params (%.3f), training bias: %s" %\
              (sum(trainable_params), sum(params), sum(trainable_params) / sum(params), "True" if train_bias else "False"))

    
    if save_lora_weights:
        torch.save(lora.lora_state_dict(model, bias=bias_str), 
                   "%s/%s" % (WEIGHTS_PATH, LORA_WEIGHTS_FIlE_NAME))
        
    return(model)

def load_model(lora_weights=True, 
               device="cpu",
               weights_path=WEIGHTS_PATH,
               weights_file=MODEL_WEIGHTS_FILE_NAME,
               lora_weights_file=LORA_WEIGHTS_FIlE_NAME):
    with torch.device(device):
        model = ESM3(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            v_heads=V_HEADS,
            n_layers=N_LAYERS,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
                
    if lora_weights:
        model = assign_lora_weights(model)
    
    state_dict = torch.load("%s/%s" % (weights_path, weights_file), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    if lora_weights:
        lora_state_dict = torch.load("%s/%s" % (weights_path, lora_weights_file), map_location=device)
        model.load_state_dict(lora_state_dict, strict=False)
    
    return (model)

@torch.enable_grad()
def inverse_folding_example():
    tokenizer = EsmSequenceTokenizer()
    tokenizers = get_model_tokenizers()
    encoder = ESM3_structure_encoder_v0("cpu")
    decoder = ESM3_structure_decoder_v0("cpu")
    model =  load_model()

    #chain = ProteinChain.from_rcsb("1utn", "A")

    # Read PDB
    chain = ProteinChain.from_pdb(WT_PATH)
    

    # Structure tokens
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cpu()
    plddt = plddt.cpu()
    residue_index = residue_index.cpu()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)
    
    # Add BOS/EOS padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097


    # Sequence tokens
    sequence = tokenizers.sequence.encode(chain.sequence)
    #sequence = tokenizer.encode('IVGGEDAGAHTRPYQVALDRGRHICGGSLINERWVVTAAHCYRDGWTAHAGEHNIRVDEGTEQRIPASKQWVHPNYDPSTLDNDIMLVKLATPATLDENVAPIPLPTVPPVEGTVCTVSGWGNTKEEGSSYPTTLQKLEVPILSDEVCQAAYPGRITPNMFCAGYLEGGKDTCQGDSGGPFVCNGELHGIVSWGEGCAQPNKPGVYTRVYLYIGWIEETIATN')


    output = model.forward(structure_coords=coords, 
                           per_res_plddt=plddt, 
                           structure_tokens=structure_tokens,
                           sequence_tokens=torch.tensor(sequence, dtype=torch.int64).reshape((1,-1))
    )
    
    
    
    sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
    function_tokens = torch.argmax(output.function_logits, dim=-1)
    sasa_tokens = torch.argmax(output.sasa_logits, dim=-1)
    ss8_tokens = torch.argmax(output.secondary_structure_logits, dim=-1)
    #residue_tokens = torch.argmax(output.residue_logits, dim=-1)
    
    sequence_delta_before_enrich = sequence_tokens.view((-1)) - torch.tensor(sequence, dtype=torch.int64).reshape((-1))
    plt.hist(sequence_delta_before_enrich)
    
    
    output2 = model.forward(structure_coords=coords,
        per_res_plddt=plddt, structure_tokens=structure_tokens,
        sequence_tokens=torch.tensor(sequence, dtype=torch.int64).reshape((1,-1)),
        function_tokens=function_tokens,
        sasa_tokens=sasa_tokens,
        #residue_annotation_tokens=residue_tokens,
        ss8_tokens=ss8_tokens)
        
    
    
    sequence_delta_after = torch.argmax(output2.sequence_logits, dim=-1).view((-1)) - torch.tensor(sequence, dtype=torch.int64).reshape((-1))
    
    plt.hist(sequence_delta_after)
    
    decoded_sequence = tokenizer.decode(sequence_tokens[0])
    print(sequence)

    

@torch.enable_grad()
def raw_forward(model,
                model_input,
                is_pdb_input=True,                                    
                input_tracks_to_complete=[],
                save_forward_pass=False,
                compute_loss=False):
    
    
    tokenizers = get_model_tokenizers()
    encoder = ESM3_structure_encoder_v0("cpu")
    model =  load_model()
    
    


def get_pdb_path_from_idx(idx):
    pass

def generate_raw_tokens_from_pdb(indices,
                                 model,
                                 structure_encoder,
                                 tokenizers,
                                 override_sequence=False):

    for idx in indices:
        chain = ProteinChain.from_pdb(get_pdb_path_from_idx(idx))
        
    
        # Structure tokens
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.cpu()
        plddt = plddt.cpu()
        residue_index = residue_index.cpu()
        _, structure_tokens = structure_encoder.encode(coords, residue_index=residue_index)
        
        # Add BOS/EOS padding
        coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
        plddt = F.pad(plddt, (1, 1), value=0)
        structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
        structure_tokens[:, 0] = BOS_STRUCTURE_TOKE
        structure_tokens[:, -1] = EOS_STRUCTURE_TOKE
    
    
        # Sequence tokens
        sequence = tokenizers.sequence.encode(chain.sequence)
        sequence_tokens=torch.tensor(sequence, dtype=torch.int64).reshape((1,-1))
                                                                          
        output = model.forward(structure_coords=coords, 
                               per_res_plddt=plddt, 
                               structure_tokens=structure_tokens,
                               sequence_tokens=sequence_tokens)
        
                
        # NEVER USE THIS
        #if override_sequence:
        #   sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
            
        function_tokens = torch.argmax(output.function_logits, dim=-1)
        sasa_tokens = torch.argmax(output.sasa_logits, dim=-1)
        ss8_tokens = torch.argmax(output.secondary_structure_logits, dim=-1)


        # ToDo Save 

def run_rosetta_job(indices):
    
    sequence_df = DesignConfiguration().df    
    aa_columns = sequence_df.columns[-N_POSITIONS_IN_DESIGN_SPACE:]
    
    all_poss = ["%sA" % p[1:] for p in aa_columns.to_list()]
    all_residues = ",".join(all_poss)
    
    for ind in indices:
        seq_job_df = sequence_df[sequence_df["sequence"] == ind]
        aa = seq_job_df[aa_columns].iloc[0,:].to_list()        
        muts = [(v[0], v[1:], i)  for i,v in enumerate(aa_columns.to_list()) if v[0] != aa[i]]
        assert len(muts) == seq_job_df["num_of_muts"].iloc[0]        
        
        mutated_residues = " ".join(["new_res%d=%s target%d=%sA" % (i+1, ONE_2_THREE[v[0]], i+1, v[1]) for i,v in enumerate(muts)])
        
        job = ""
        job += "%s" % ROSETTA_SCRIPTS
        job += "-database %s " % ROSETTA_DB
        job += "@{%s}/initial_data/flags " % ROSETTA_FILES_PATH
        job += "-out:prefix %s_" % ind
        job += "-out:file:scorefile %s.sc " % ind
        job += "-parser:script_vars "
        job += "all_ress=%s " % all_residues
        job += "%s" % mutated_residues
        
        process = subprocess.run(job, shell=True, check=True, text=True, capture_output=True)


def generate_data_for_pseudo_protein(indices):
    
    run_rosetta_jobs(indices)
        
    
    model = load_model()
    tokenizers = get_model_tokenizers()
    encoder = ESM3_structure_encoder_v0("cpu")
    
    generate_raw_tokens_from_pdb(indices, 
                                 model,
                                 encoder,
                                 tokenizers)
    
    
    
   

inverse_folding_example()
#def master():
    
 
raw_forward(model, model_input)
    

    
args = parser.parse_args()

if args.mode == "master":
    scheduler_main()
elif args.mode == "rosetta":
    pass
elif args.mode == "forward":
    pass
elif args.mode == "gen_pdb":
    pass
elif args.mode == "grad_master":
    pass
elif args.mode == "grad":
    pass
    
# Run the command


