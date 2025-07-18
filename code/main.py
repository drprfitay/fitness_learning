#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 23:43:09 2024

@author: itayta
"""

import huggingface_hub
import sys, os
import random
import torch
import torch.nn.functional as F
import loralib as lora
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import argparse
import re
import shutil
import pickle
import threading

from multiprocessing import Process, current_process

parser = argparse.ArgumentParser(description="ItayFold main")


parser.add_argument("--is_sequence_indices", type=bool, help="Flag to determine whether indices are numerical or full sequences (pseudo sequeces)", required=False, default=False)
parser.add_argument("--mode", type=str, help="'scheduler' or 'worker'", required=False, default=None)
parser.add_argument("--indices", nargs='+', help='List of indices to work on')
parser.add_argument("--index_range", nargs='+', help='Range of indices to work on', default="")
parser.add_argument("--operations", nargs='+', help='List of operations to perform', default=[])
parser.add_argument("--override", type=bool, help="Override generations", required=False, default=False)
parser.add_argument("--send_to_cluster", type=bool, help="Sends entire operation to cluster", required=False, default=False)
parser.add_argument("--seq_space_file", type=str, help='Path to designed sequence space for splitting train/test, Should be a .pkl file', required=False)
parser.add_argument("--is_seq_space_full_path", type=bool, help="Specifies whether 'seq_space_path' is full path or relative to sequence spaces folder (SEQ_SPACES_PATH in constants)", required=False, default=False)
parser.add_argument("--scheduler", type=str, help="Specifies the dataset to operate on (SEQ_SPACES_PATH in constants)", required=False, default=None)
parser.add_argument("--run_statistics", type=str, help="Specifies the dataset to run statistics on (SEQ_SPACES_PATH in constants); similar to scheduling", required=False, default=None)
parser.add_argument("--zip", type=str, help="Specifies the dataset to zip (SEQ_SPACES_PATH in constants); similar to scheduling", required=False, default=None)
parser.add_argument("--zipformat", type=str, help="Specifies what to pack in zip, 'T' Tensors, 'E' embeddings, 'P' pdbs, 'R' rosetta energies. Example usage --zipformat TE", required=False, default=None)
parser.add_argument("--execute_missing_sequences", type=str, help="When scheduling, specifies whether to use missing sequences that were not generated with respect to 'seq_space_path'. 'T' Tensors, 'E' embeddings, 'P' pdbs, 'R' rosetta energies.", required=False, default=None)
parser.add_argument("--test_dataset", type=bool, help="When scheduling, specifies whether operating on test dataset and not train", required=False, default=False)
parser.add_argument("--njobs", type=int, help="When scheduling, specifies number of jobs ", required=False, default=100)
parser.add_argument("--setup", type=bool, help="Sets up the environment and creates directories", required=False, default=False)
parser.add_argument("--lt", type=bool, help="Specifies whether using the long term storages or not", required=False, default=False)
parser.add_argument("--rsynclt", type=bool, help="Specifies whether to sync between long term and regulat storage", required=False, default=False)
parser.add_argument("--rsynclt_remove_files", type=bool, help="Specifies whether to remove files after sync", required=False, default=False)
parser.add_argument("--Nthreads", type=int, help="Specifies the number of threads to use when possible", required=False, default=1)
 

#parser.add_argument("--gen_", type=str, help="Specifies the train dataset to operate on (SEQ_SPACES_PATH in constants)", required=False, default=None)


# Parse the arguments
args = parser.parse_args()

from constants import *
from utils import *

fix_esm_path()

import esm


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

from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
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

if USE_HF:
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

def load_structure_encoder(device="cpu",
                           weights_path=WEIGHTS_PATH,
                           weights_file=ENCODER_WEIGHTS_FILE_NAME):
    with torch.device(device):
        model = StructureTokenEncoder(d_model=ENCODER_D_MODEL, 
                                      n_heads=ENCODER_N_HEADS, 
                                      v_heads=ENCODER_V_HEADS, 
                                      n_layers=ENCODER_N_LAYERS, 
                                      d_out=ENCODER_D_OUT, 
                                      n_codes=ENCODER_N_CODES).eval()
        
    state_dict = torch.load("%s/%s" % (weights_path, weights_file), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    return model

def load_structure_decoder(device="cpu",
                           weights_path=WEIGHTS_PATH,
                           weights_file=DECODER_WEIGHTS_FILE_NAME):
    
    with torch.device(device):
        model = StructureTokenDecoder(d_model=DECODER_D_MODEL, 
                                      n_heads=DECODER_N_HEADS, 
                                      n_layers=DECODER_N_LAYERS).eval()
        
    state_dict = torch.load("%s/%s" % (weights_path, weights_file), map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    return model    

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
    chain = ProteinChain.from_pdb(WT_PDB_PATH)
    

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
                           sequence_tokens=torch.tensor(sequence, dtype=torch.int64).reshape((1,-1)),
                           return_embeddings=True)
    
    output, all_embeddings = output
    
    
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


@torch.no_grad()
def generate_embeddings_from_model(indices,
                                   model,
                                   model_name,
                                   is_sequence_indices,
                                   override=False,
                                   sanity_check=True):

    if not is_sequence_indices:
        indices = [int(i) for i in indices]
        
    print(indices)
    
    save_path = "%s/%s" % (MODEL_EMBEDDINGS_PATH, model_name)
    os.makedirs(save_path, exist_ok=True)

    for idx in indices:
        sequence_df = DesignConfiguration().df    
        uid_column = "sequence" if is_sequence_indices else "idx"
        seq_job_df = sequence_df[sequence_df[uid_column] == idx]
        seq_str = seq_job_df["sequence"].iloc[0]

        token_tensor_file_name = "tokens_%s.pth" % seq_str
        if not file_in_dir(token_tensor_file_name, RAW_TENSORS_PATH):
            if VERBOSE:
                print("Raw tokens file %s not generated" % token_tensor_file_name)
            continue

        
        embeddings_file_name = "embeddings_%s.pth" % seq_str
        if not override and file_in_dir(embeddings_file_name, save_path):
            if VERBOSE:
                print("Raw embeddings file %s already generated" % token_tensor_file_name)
            continue
            
    
        raw_tokens = torch.load("%s/%s" % (RAW_TENSORS_PATH, token_tensor_file_name))
        
        output = model.forward(structure_coords=raw_tokens["coords"], 
                               per_res_plddt=raw_tokens["plddt"], 
                               structure_tokens=raw_tokens["structure_tokens"],
                               sequence_tokens=raw_tokens["sequences_tokens"],
                               sasa_tokens=raw_tokens["sasa_tokens"],
                               ss8_tokens=raw_tokens["ss8_tokens"],
                               function_tokens=raw_tokens["function_tokens"],
                               return_embeddings = True)
        
        
        final_embeddings = {"normed_emb_last_layer": output[0].embeddings,
                            "all_emb": output[1]}

        torch.save(final_embeddings, "%s/%s" % (save_path, embeddings_file_name))    




@torch.no_grad()
def generate_raw_tokens_from_pdb(indices,
                                 model,
                                 structure_encoder,
                                 tokenizers,
                                 is_sequence_indices,
                                 override=False,
                                 merge_energies=True,
                                 override_sequence=False,
                                 override_structure=False,
                                 sanity_check=True):

    if not is_sequence_indices:
        indices = [int(i) for i in indices]
    print(indices)


    for idx in indices:
        sequence_df = DesignConfiguration().df    
        uid_column = "sequence" if is_sequence_indices else "idx"
        seq_job_df = sequence_df[sequence_df[uid_column] == idx]
        seq_str = seq_job_df["sequence"].iloc[0]
        pdb_file_name = "%s_refined.pdb" % seq_str

        if not file_in_dir(pdb_file_name, PDB_ROSETTA_SCORES_PATH):
            if VERBOSE:
                print("%s does not exist!" % pdb_file_name)
                print("Continuing to next sequence")
            continue


        token_tensor_file_name = "tokens_%s.pth" % seq_str
        if not override and file_in_dir(token_tensor_file_name, RAW_TENSORS_PATH):
            if VERBOSE:
                print("Raw tokens file %s already generated" % token_tensor_file_name)
            continue
            
    
        chain = ProteinChain.from_pdb("%s/%s" % (PDB_ROSETTA_SCORES_PATH, pdb_file_name))
        
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
        structure_tokens[:, 0] = BOS_STRUCTURE_TOKEN
        structure_tokens[:, -1] = EOS_STRUCTURE_TOKEN
    
    
        # Sequence tokens
        sequence = tokenizers.sequence.encode(chain.sequence)
        sequence_tokens=torch.tensor(sequence, dtype=torch.int64).reshape((1,-1))
                                                                          
        output = model.forward(structure_coords=coords, 
                               per_res_plddt=plddt, 
                               structure_tokens=structure_tokens,
                               sequence_tokens=sequence_tokens)
        
            
        function_tokens = torch.argmax(output.function_logits, dim=-1)
        sasa_tokens = torch.argmax(output.sasa_logits, dim=-1)
        ss8_tokens = torch.argmax(output.secondary_structure_logits, dim=-1)
        
        # NEVER USE THIS
        # if override_sequence:
        #     if VERBOSE:
        #         print("Overriding sequence tokens!")
        #     sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)

        if override_structure:
            if VERBOSE:
                print("Overriding structure_tokens tokens!")
            structure_tokens = torch.argmax(output.structure_logits, dim=-1)


        tokens_tensor = {"sequences_tokens": sequence_tokens,
                         "function_tokens": function_tokens,
                         "sasa_tokens": sasa_tokens,
                         "ss8_tokens": ss8_tokens,                         
                         "structure_tokens": structure_tokens,
                         "plddt":plddt,
                         "coords":coords,
                         "residue_index": residue_index,
                         "activity_labels_values": seq_job_df.iloc[:,SEQ_DF_ACTIVITY_LABEL_START_IDX:SEQ_DF_ACTIVITY_LABEL_END_IDX].to_numpy(),
                         "activity_labels": seq_job_df.columns[SEQ_DF_ACTIVITY_LABEL_START_IDX:SEQ_DF_ACTIVITY_LABEL_END_IDX].to_numpy()}
        # ToDo Save
        energy_matrix_file_name =  "energies_%s.pth" % seq_str
        if merge_energies and file_in_dir(energy_matrix_file_name, RAW_TENSORS_ENERGIES_PATH): 
            if VERBOSE:
                print("Found energy matrix file %s" % energy_matrix_file_name)

            loaded_energy_tensors = torch.load("%s/%s" % (RAW_TENSORS_ENERGIES_PATH, energy_matrix_file_name))
            tokens_tensor["energy"] = loaded_energy_tensors["energy"]
            tokens_tensor["energy_labels"] = loaded_energy_tensors["labels"]


        if VERBOSE:
            print("Saving tokens_tensor %s at %s" % (token_tensor_file_name, RAW_TENSORS_PATH))

        torch.save(tokens_tensor, "%s/%s" % (RAW_TENSORS_PATH, token_tensor_file_name))    

        if sanity_check:
            input_sequence_tokens = tokenizers.sequence.encode(chain.sequence)
            output_sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
            delta = output_sequence_tokens.view((-1)) - torch.tensor(input_sequence_tokens, dtype=torch.int64).reshape((-1))
            decoded_output_sequence = tokenizers.sequence.decode(output_sequence_tokens[0])  
            decoded_input_sequence = tokenizers.sequence.decode(input_sequence_tokens)
            
            print(output_sequence_tokens.shape)
            print(torch.tensor(input_sequence_tokens, dtype=torch.int64).reshape((-1)).shape)
            print(structure_tokens.shape)
            print(sasa_tokens.shape)
            print(delta)
            print("############ DECODED INPUT ###########")
            print("\t%s" % decoded_input_sequence)
            print("############ DECODED OUTPUT ###########")
            print("\t%s" % decoded_output_sequence)
            

def run_rosetta_job(indices, is_sequence_indices, gen_pdb=True, override=False, rosetta_job_generator=None, extract_energies=True):
    jobs_to_run, pdb_files = rosetta_job_generator(indices, is_sequence_indices, gen_pdb, override)

    for idx, job in enumerate(jobs_to_run):
        process = subprocess.run(job, shell=True, check=False, text=True, capture_output=True)

        if process.returncode != 0:
            executed_sequence = pdb_files[idx].split("_refined.pdb")[0]

            if VERBOSE:
                print("Error (%s) %d \{ %s \}" % (executed_sequence, process.returncode, process.stderr))

            append_to_log(executed_sequence, 
                          "Error %d \{ %s \}" % (process.returncode, process.stderr),
                          job)

    if extract_energies:
        for seq, pdb_full_path in pdb_files:

            energy_matrix_file_name =  "energies_%s.pth" % seq

            if not override and file_in_dir(energy_matrix_file_name, RAW_TENSORS_ENERGIES_PATH):
                if VERBOSE:
                    print("Energy file for %s already generated" % energy_matrix_file_name)
                continue

            energy_matrix, labels = parse_energys_matrix_from_pdb(pdb_full_path)
            energy_matrix = torch.tensor(energy_matrix)
            energy_dict = {"energy":energy_matrix, "labels":labels}

            if VERBOSE:
                print("Generating energies for %s (shape %s)" % (seq, str(energy_matrix.shape)))

            torch.save(energy_dict, "%s/%s" % (RAW_TENSORS_ENERGIES_PATH, energy_matrix_file_name))
    
def generate_data_for_pseudo_protein(indices):
    
    run_rosetta_jobs(indices)
        
    
    model = load_model()
    tokenizers = get_model_tokenizers()
    encoder = ESM3_structure_encoder_v0("cpu")
    
    generate_raw_tokens_from_pdb(indices, 
                                 model,
                                 encoder,
                                 tokenizers)
    
            
def splitter(seq_space_file=None,
             is_seq_space_full_path=False):

    BSUB_ESM_EXEC_PAYLOAD = """
    #!/bin/bash
    conda activate esm_env
    cd %s
    python %s %s
    """#  % (CODE_PATH, MAIN_JOB_MASTER_FILE_NAME)

    #Todo:
    # selected_sec_space = Read from selected_seq_space_path
    # selected_sec_space = {'42A': ['L'],\
    #      '44A': ['L', 'I'],
    #      '46A': ['F'],
    #      '61A': ['V'],
    #      '64A': ['L'],
    #      '68A': ['V', 'A', 'M'],
    #      '69A': ['Q', 'Y', 'L', 'M'],
    #      '110A': ['A'],
    #      '112A': ['V', 'I'],
    #      '145A': ['Y', 'F'],
    #      '150A': ['V', 'I'],
    #      '163A': ['V'],
    #      '165A': ['F'],
    #      '167A': ['T'],
    #      '181A': ['H', 'V', 'I', 'Y'],
    #      '201A': ['L'],
    #      '220A': ['L'],
    #      '224A': ['V'],
    #      '14A': ['I'],
    #      '16A': ['V'],
    #      '18A': ['L'],
    #      '72A': ['S', 'V', 'T', 'C', 'A'],
    #      '108A': ['T', 'V', 'I', 'L', 'E'],
    #      '119A': ['L'],
    #      '123A': ['I', 'V'],
    #      '185A': ['N', 'V']}

    if not seq_space_file.endswith(".pkl"):
        print("Sequence space file flag '--seq_space_file' is not a .pkl file. Provided %s" % seq_space_file)
        return()

    if not is_seq_space_full_path:
        full_seq_space_file = "%s/%s" % (SEQ_SPACES_PATH, seq_space_file)

    if VERBOSE:
        print("Reading seq space from %s " % seq_space_file)

    with open(full_seq_space_file, "rb") as file:
        selected_seq_space = pickle.load(file)

    sequence_df = DesignConfiguration().df   

    keys =  list(selected_seq_space.keys())
    selected_seq_space = dict([(key, selected_seq_space[key]) for key in keys])


    train_df = sequence_df.copy()
    test_df = sequence_df.copy()
    train_indices = np.repeat([True], sequence_df.shape[0])

    for k,v in selected_seq_space.items():

        work_col = "%s%d" % (v[0], int(k[0:-1]))
        
        if np.isin(work_col, sequence_df.columns) == True: 
            pass
        else:
            if VERBOSE:
                print("%s is not a mutation used in seq space" % work_col)
            continue
        
        train_indices = train_indices & np.isin(sequence_df[work_col], v)
    
    test_indices = ~train_indices
    train_df = train_df.iloc[train_indices,]
    test_df = test_df.iloc[test_indices]

    if VERBOSE:
        print("Split sequences df (%s) to" % (str(sequence_df.shape)))
        print("\t Train: (%s)" % (str(train_df.shape)))
        print("\t Test: (%s)" % (str(test_df.shape)))
        print("\t Overall %d sequences:" % (train_df.shape[0] + test_df.shape[0]))


    test_train_split_csv_name = seq_space_file.split(".pkl")[0]

    if VERBOSE:
        print("Saving splits into as %s_train %s_test in %s" % (test_train_split_csv_name, test_train_split_csv_name, TRAIN_TEST_SPLITS_PATH))

    train_csv = train_df.loc[:, ["idx", "sequence"]]
    test_csv = test_df.loc[:, ["idx", "sequence"]]
    train_csv.to_csv("%s/%s_train.csv" % (TRAIN_TEST_SPLITS_PATH, test_train_split_csv_name))
    test_csv.to_csv("%s/%s_test.csv" % (TRAIN_TEST_SPLITS_PATH, test_train_split_csv_name))
        


args = parser.parse_args()
    

def do_gen_embeddings():
    model =  load_model()
    prefix = MODEL_WEIGHTS_FILE_NAME.split(".pth")[0]
    
    generate_embeddings_from_model(args.indices,
                                   model,
                                   prefix,
                                   args.is_sequence_indices,
                                   override=args.override)
# Run the command
def do_gen_tokens():
    model =  load_model()
    tokenizers = get_model_tokenizers()
    structure_encoder = load_structure_encoder()

    generate_raw_tokens_from_pdb(args.indices,
                                 model,
                                 structure_encoder,
                                 tokenizers,
                                 args.is_sequence_indices,
                                 override=args.override,
                                 sanity_check=False)


def do_rosetta():
    run_rosetta_job(args.indices, 
                    args.is_sequence_indices,                     
                    gen_pdb=True, 
                    override=args.override,
                    rosetta_job_generator=gfp_rosetta_job_generator)     

def do_splitter():
    splitter(args.seq_space_file,
             args.is_seq_space_full_path)



supported_modes = {'splitter': do_splitter,
                   'rosetta': do_rosetta,
                   'gen_tokens': do_gen_tokens,
                   'gen_embeddings': do_gen_embeddings}
# def do_all(is_master, mode):
#     if is_master == "master":
#         pass

#     elif mode == "splitter":
#         do_splitter()

#     elif mode == "rosetta":
#         do_rosetta()
#     elif mode == "forward":
#         pass
#     elif mode == "gen_tokens":
#         do_gen_tokens()

#     elif mode == "gen_pdb":
#         pass    
#     elif mode == "grad_master":
#         pass
#     elif mode == "grad":
#         pass


def send_job_to_cluster():

    provided_args = {key: value for key, value in vars(args).items() if parser.get_default(key) != value and key != 'send_to_cluster'}
    
    forwarded_args = " ".join(["--%s %s" % (k, " ".join(v) if type(v) == list else v) for k,v in provided_args.items()])

    random_seed_1 = random.randint(1000, 10000000)
    random_seed_2 = random.randint(1000, 10000000)
    random_exec_file_name = "%s/execution_%d_%d.sh" % (CLUSTER_EXECUTIONS_PATH, random_seed_1, random_seed_2)

    cluster_bash_file_exec_payload = """
    #!/bin/bash
    conda init
    conda activate esm_env
    echo 'Removing %s'
    rm %s
    echo 'Moving to %s to execute'
    cd %s
    python main.py %s
    """ % (random_exec_file_name, 
         random_exec_file_name, 
         CODE_PATH, 
         CODE_PATH, 
         forwarded_args)

    with open(random_exec_file_name, 'w') as file:
        file.write(cluster_bash_file_exec_payload)
        file.close()

    node_command = BSUB_COMMAND % (LONG_Q,
                                   MEM,
                                   "%s/%d_%d_outfile" % (CLUSTER_EXECUTIONS_PATH, random_seed_1, random_seed_2),
                                    "%s/%d_%d_errfile" % (CLUSTER_EXECUTIONS_PATH, random_seed_1, random_seed_2),
                                    "exec /%s" % random_exec_file_name)

    if VERBOSE:
        print(node_command)

    process = subprocess.run(node_command, shell=True, check=True, text=True, capture_output=True)   

if args.rsynclt:
    pass

if args.setup:
    paths_to_create  = [DATA_PATH,
                        CODE_PATH,
                        LOG_PATH,
                        CLUSTER_EXECUTIONS_PATH,
                        CONFIGURATION_PATH,
                        PDB_ROSETTA_SCORES_PATH,
                        ROSETTA_SCORES_PATH,
                        RAW_TENSORS_PATH,
                        RAW_TENSORS_ENERGIES_PATH,
                        TRAIN_TEST_SPLITS_PATH,
                        SEQ_SPACES_PATH,
                        ZIP_PATH,
                        DATASETS_PATH,
                        MODEL_EMBEDDINGS_PATH]

    for p in paths_to_create:
        os.makedirs(p, exist_ok=True)
    exit()
   
# if args.gen_dataset: is not None:

if args.zip is not None:

    if args.is_seq_space_full_path:
        print("Please specify the sequence space in SEQ_SPACE folder and do not provide full path when zipping")
        exit()
    
    
    results = get_missing_sequences(args.is_seq_space_full_path, 
                                    args.zip, 
                                    args.test_dataset, 
                                    return_full_path = True, 
                                    model_name=MODEL_WEIGHTS_FILE_NAME.split(".pth")[0],
                                    long_term=args.lt)
    
    pdb_generated = results["pdb_generated"]
    tokens_generated = results["tokens_generated"]
    energies_generated = results["energies_generated"]
    embeddings_generated = results["embeddings_generated"]
    all_energy_files = results["all_energy_files"]
    all_pdb_files = results["all_pdb_files"]
    all_esm_tensor_files = results["all_esm_tensor_files"]
    all_embeddings_generateed = results["all_embeddings_generated"]

    pdbs = np.array(all_pdb_files)[np.isin(all_pdb_files, pdb_generated)]
    esm_tokens = np.array(all_esm_tensor_files)[np.isin(all_esm_tensor_files, tokens_generated)]
    energy_tensors = np.array(all_energy_files)[np.isin(all_energy_files, energies_generated)]
    energy_tensors = np.array(all_energy_files)[np.isin(all_energy_files, energies_generated)]
    embeddings = np.array(all_embeddings_generateed)[np.isin(all_embeddings_generateed, embeddings_generated)]

    if VERBOSE:
        print("Zipping up dataset %s files included: " % args.zip)
        print("\t %d pdb files " % len(pdbs))
        print("\t %d esm token files " % len(esm_tokens))
        print("\t %d rosetta energy files " % len(energy_tensors))
        print("\t %d embedding files " % len(embeddings))


    if args.lt:
        zip_path_to_use = LONG_TERM_STORAGE_ZIP_PATH
    else: 
        zip_path_to_use = ZIP_PATH

    work_zip_path = "%s/%s_%s_zipped/" % (zip_path_to_use,
                                        "test" if args.test_dataset else "train",
                                         args.zip)
    os.makedirs(work_zip_path, exist_ok = True)
    tmp_directory = "%s/files" % work_zip_path

    if os.path.exists(tmp_directory) and args.override:
        shutil.rmtree(tmp_directory)

    os.makedirs(tmp_directory, exist_ok = True)

    if args.test_dataset:
        csv_file_path = "%s/%s_test.csv" % (TRAIN_TEST_SPLITS_PATH, args.zip)
    else:
        csv_file_path = "%s/%s_train.csv" % (TRAIN_TEST_SPLITS_PATH, args.zip)
        
    shutil.copy(csv_file_path, "%s/sequences.csv" % tmp_directory)

    
    if args.zipformat is None:
        files_dict = {"pdbs":pdbs,
                    "tokens":esm_tokens,
                    "energies":energy_tensors,
                    "embeddings":embeddings}
    else: 
        files_dict = {}

        zip_format_chars = [c for c in args.zipformat]

        things_to_add = {'T': ("Tensors/Tokens",  "tokens", esm_tokens),
                         'R': ("Rosetta energies", "energies", energy_tensors),
                         'E': ("Embeddings", "embeddings", embeddings),
                         'P': ("PDB files", "pdbs", pdbs)}
                         
        for k, item in things_to_add.items():
            if k in zip_format_chars:
                print_str, res_dir, list_of_files = item
                if VERBOSE:
                    print("Adding '%s' to zip file" % print_str)

                files_dict[res_dir] = list_of_files
                 

    threads = {}

    def copy_files(tmp_work_path, v, td=None):
        for idx, full_path_file in enumerate(v):
            if VERBOSE and idx % 1000 == 0:
                print("\t%sDone copying (%d / %d)" % ("" if td == None else "(%d) " % td, idx, len(np.unique(v))))

            if not file_in_dir(full_path_file.split("/")[-1], tmp_work_path):
                shutil.copy(full_path_file, tmp_work_path)
                
            elif VERBOSE and idx % 1000 == 0:
                print("\t%sSkipping (%d / %d)" % ("" if td == None else "(%d) " % td, idx, len(np.unique(v))))

        # if td is not None:
        #     thrd_name = current_process().name
        #     threads[thrd_name].join()


    for k,v in files_dict.items():
        tmp_work_path = "%s/%s/" % (tmp_directory, k)
        os.makedirs(tmp_work_path, exist_ok = True)
        
        
        if VERBOSE:
            print("Zipping up %s" % k)
        
        N_threads = args.Nthreads

        # if N_threads > 1:
        #     if VERBOSE:
        #         print("Running %d threads for zip" % N_threads  )
        #     chunk_size = len(v) // N_threads

            
        #     for td in range(N_threads):
        #         print("Starting thread %d for zipping of %s" % (td, k))
        #         si = td * chunk_size
        #         if td == (N_threads - 1):                    
        #             fi = len(v)
        #         else:
        #             fi = (td + 1) * chunk_size

        #         thrd_name = "td%d" % td
        #         thrd = Process(target=copy_files, args=(tmp_work_path, v[si:fi], td, ), name=thrd_name)
        #         threads[thrd_name] = thrd
        #         thrd.start()
            
        #     for td in range(N_threads):
        #         thrd_name = "td%d" % td
        #         threads[thrd_name].join()
        #         if VERBOSE:
        #             print("Zip thread %d is finished" % td)

        print(len(np.unique(v)))

        print(v[1:100])

        fast_file_copy_or_move(from_dir="/",
                               to_dir=tmp_work_path,
                               file_list=v,
                               blocking=True)
                               
        print(tmp_work_path)

        # for full_path_file in v:
        #     #shutil.copy(full_path_file, tmp_work_path)
            


                

    # shutil.make_archive("%s/%s_%s" % (work_zip_path, 
    #                                   "test" if args.test_dataset else "train", 
    #                                   args.zip), 
    #                     'zip', 
    #                     tmp_directory)

    zip_file_name = "%s/%s_%s.tar.zst" % (work_zip_path, "test" if args.test_dataset else "train", args.zip)
    fast_zip(tmp_directory, zip_file_name, blocking=True)
    
    if VERBOSE:
        print("Removing tmp directory %s" % tmp_directory)
    #shutil.rmtree(tmp_directory)
    exit()
 
if args.run_statistics is not None:
    results = get_missing_sequences(args.is_seq_space_full_path, 
                                    args.run_statistics, 
                                    is_test=args.test_dataset,
                                    model_name=MODEL_WEIGHTS_FILE_NAME.split(".pth")[0],
                                    long_term=args.lt)

    all_energy_files = results["all_energy_files"]
    all_pdb_files = results["all_pdb_files"]
    all_esm_tensor_files = results["all_esm_tensor_files"]
    sequences_in_set = results["sequences_in_set"]
    pdb_generated = results["pdb_generated"]
    tokens_generated = results["tokens_generated"]
    energies_generated = results["energies_generated"]
    embeddings_generated = results["embeddings_generated"]
    all_embeddings_generated = results["all_embeddings_generated"]

    n_pdbs = np.isin(all_pdb_files, pdb_generated)
    n_esm_tokens = np.isin(all_esm_tensor_files, tokens_generated)
    n_energy_tensors = np.isin(all_energy_files, energies_generated)
    n_embeddings = np.isin(all_embeddings_generated, embeddings_generated)

    
    print("####### SUMMARY STATISTICS OVER %s" % args.run_statistics)
    print("\tOverall: %d/%d (%.3f) energy files generated" % (np.sum(n_energy_tensors), len(sequences_in_set), np.sum(n_energy_tensors)/len(sequences_in_set)))
    print("\tOverall: %d/%d (%.3f) pdb files generated" % (np.sum(n_pdbs), len(sequences_in_set), np.sum(n_pdbs)/len(sequences_in_set)))
    print("\tOverall: %d/%d (%.3f) esm token files generated" % (np.sum(n_esm_tokens), len(sequences_in_set), np.sum(n_esm_tokens)/len(sequences_in_set)))
    print("\tOverall: %d/%d (%.3f) embeddings generated" % (np.sum(n_embeddings), len(sequences_in_set), np.sum(n_embeddings)/len(sequences_in_set)))

    exit()

if args.scheduler is not None:

    if not args.is_seq_space_full_path:
        full_seq_space_file = "%s/%s" % (TRAIN_TEST_SPLITS_PATH, args.scheduler)

    if VERBOSE:
        print("Scheduling over %s" % args.scheduler)
    
    if args.operations == [] and args.mode is None :
        print("Please specify what to do when running scheduler specify --args.operations or --args.mode")

    if args.operations != []:
        executions = args.operations
    else:
        executions = [args.mode]

    print(executions)
    executions = " ".join(executions)

    if args.test_dataset:
        scheduling_df = pd.read_csv("%s_test.csv" % full_seq_space_file)
    else:
        scheduling_df = pd.read_csv("%s_train.csv" % full_seq_space_file)

    

    if args.execute_missing_sequences is not None:

        if args.execute_missing_sequences == "E":
            esm_model_name = MODEL_WEIGHTS_FILE_NAME.split(".pth")[0]
        else:
            esm_model_name = None
        
        results = get_missing_sequences(args.is_seq_space_full_path, 
                                        args.scheduler, 
                                        is_test=args.test_dataset, 
                                        return_full_path=False,
                                        model_name=esm_model_name)

        sequences_in_set = results["sequences_in_set"]

        #TODO aggregate 

        #missing_sequences = np.array([True] * )
        if args.execute_missing_sequences == "R":
            generated = results["energies_generated"]
            all_files = results["all_energy_files"]        
        elif args.execute_missing_sequences == "T":            
            generated = results["tokens_generated"]
            all_files = results["all_esm_tensor_files"]
        elif args.execute_missing_sequences == "E":                        
            generated = results["embeddings_generated"]
            all_files = results["all_embeddings_generated"]
        else:
            generated = results["pdb_generated"]
            all_files = results["all_pdb_files"]        
        
        print(all_files)
        print(generated)
        print(len(generated))
        print(len(all_files))
        missing_sequences = ~np.isin(all_files, generated)
        if VERBOSE:
            print("There are %d sequences missing out of %d" % (sum(missing_sequences), len(sequences_in_set)))

        scheduling_df = scheduling_df.loc[missing_sequences,:]


    n_jobs = args.njobs
    n_sequences = scheduling_df.shape[0]
    sequences_in_job =n_sequences // n_jobs
    for job_idx in range(0, n_jobs):
        
        if job_idx == n_jobs - 1: # last job
            indices = scheduling_df.iloc[(job_idx * sequences_in_job):(n_sequences),:]["idx"].to_list()
        else:
            indices = scheduling_df.iloc[(job_idx * sequences_in_job):((job_idx + 1)*sequences_in_job),:]["idx"].to_list()
        indices_str = " ".join([str(idx) for idx in indices])
    
        command = "python %s --operations %s --indices %s --send_to_cluster True" % (MAIN_JOB_MASTER_FILE_NAME, executions, indices_str)

        if VERBOSE:
            print("Scheduler mode running command :")
            print("\t%s" % command)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
    
    exit()

if args.send_to_cluster:
    send_job_to_cluster()
    exit()

if args.operations != []:
    executions = args.operations  
else: 
    executions = [args.mode]

for mode in executions:
    if mode not in supported_modes.keys():
        print("Provided mode %s is not currently supported" % mode)
        print("Currently supported modes are:")
        print("".join(["\t%s\n" % k for k in supported_modes.keys()]))
        exit()

for mode in executions:
    if VERBOSE:
        print("Running mode %s" % mode)
    execution = supported_modes[mode]
    execution()
    





    
#do_all(args.master, args.mode)