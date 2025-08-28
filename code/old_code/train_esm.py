# -*- coding: utf-8 -*-

import sys, os
import torch
import torch.nn.functional as F
import loralib as lora
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from esm_smart_dataset import EsmSequenceActivityDataset, Esm2SequenceActivityDataset
from sequence_space_utils import *

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices

blosum62 = substitution_matrices.load("BLOSUM62")


ROOT_PATH = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
RESULTS_PATH = "%s/results" % ROOT_PATH
ITAYFOLD_PATH = "%s/itayFold/" % ROOT_PATH
WEIGHTS_PATH = "/%s/weights/" % ITAYFOLD_PATH

MODEL_WEIGHTS_FILE_NAME = "esm3/esm_model_weights.pth"
LORA_WEIGHTS_FIlE_NAME =  "esm3/esm_lora_weights.pth"
ENCODER_WEIGHTS_FILE_NAME = "esm3/structure_encoder.pth"
DECODER_WEIGHTS_FILE_NAME = "esm3/structure_decoder.pth"



ROOT_DMS_PATH = "%s/data/datasets/DMS/Data" % ROOT_PATH
BASE_DMS_PATH = "%s/data/" % ROOT_DMS_PATH
BASE_DMS_PDB_PATH = "%s/structure_data/" % ROOT_DMS_PATH 



dms_datasets_to_exclude = ["BLAT_ECOLX_Ranganathan2015-2500", ".DS_Store",
                           "BLAT_ECOLX_Ostermeier2014-linear",
                           "YAP1_HUMAN_Fields2012-singles-linear",
                           "GFP_AEQVI_Sarkisyan2016",
                           "UBE4B_MOUSE_Klevit2013-nscor_log2_ratio_single_NMR",
                           "UBE4B_MOUSE_Klevit2013-nscor_log2_ratio_single",
                           "DLG4_RAT_Ranganathan2012-CRIPT",
                           "PABP_YEAST_Fields2013-linear",
                           "GB1_Olson2014_ddg",
                           "BLAT_ECOLX_Tenaillon2013-singles-MIC_score",
                           "BLAT_ECOLX_Palzkill2012-ddG_stat",
                           "B3VI55_LIPST_Klesmith2015_SelectionOne",
                           "B3VI55_LIPSTSTABLE_Klesmith2015_SelectionTwo"]
    
    
all_dms_datasets = os.listdir(BASE_DMS_PATH)
all_dms_datasets = [ds for ds in all_dms_datasets if ds not in dms_datasets_to_exclude]


ENCODER_D_MODEL=1024
ENCODER_N_HEADS=1
ENCODER_V_HEADS=128
ENCODER_N_LAYERS=2
ENCODER_D_OUT=128
ENCODER_N_CODES=4096

DECODER_D_MODEL=1280
DECODER_N_HEADS=20
DECODER_N_LAYERS=30

D_MODEL = 1536
N_LAYERS = 48
N_HEADS = 24
V_HEADS = 256
LORA_R = 16

BOS_STRUCTURE_TOKEN = 4098
EOS_STRUCTURE_TOKEN = 4097 

LORA_SEQUENCE_HEAD = True
LORA_EMBEDDINGS = True
LORA_TRANSFORMER_LINEAR_WEIGHTS = True
LORA_OUTPUT_HEADS = True

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

from esm.models.esm3 import ESM3
from esm.tokenization import get_model_tokenizers
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants import esm3 as C


# from esm.utils.constants.models import (
#     ESM3_FUNCTION_DECODER_V0,
#     ESM3_OPEN_SMALL,
#     ESM3_STRUCTURE_DECODER_V0,
#     ESM3_STRUCTURE_ENCODER_V0,
# )

# from esm.pretrained import (
#     ESM3_function_decoder_v0,
#     ESM3_sm_open_v0,
#     ESM3_structure_decoder_v0,
#     ESM3_structure_encoder_v0,
# )

from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)

#from esm.tokenization import get_esm3_model_tokenizers
# from esm.tokenization.function_tokenizer import (
#     InterProQuantizedTokenizer as EsmFunctionTokenizer,
# )

# from esm.tokenization.sequence_tokenizer import (
#     EsmSequenceTokenizer,
# )

#from esm.utils.structure.protein_chain import ProteinChain
#from esm.utils.types import FunctionAnnotation

import esm2

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
                    
    if LORA_EMBEDDINGS:
        model.encoder.sequence_embed = \
            lora.Embedding(model.encoder.sequence_embed.num_embeddings, 
                           model.encoder.sequence_embed.embedding_dim,
                           LORA_R)
        model.encoder.structure_tokens_embed = \
            lora.Embedding(model.encoder.structure_tokens_embed.num_embeddings, 
                           model.encoder.structure_tokens_embed.embedding_dim,
                           LORA_R)
        model.encoder.ss8_embed = \
            lora.Embedding(model.encoder.ss8_embed.num_embeddings, 
                           model.encoder.ss8_embed.embedding_dim,
                           LORA_R)
        model.encoder.sasa_embed = \
            lora.Embedding(model.encoder.sasa_embed.num_embeddings, 
                           model.encoder.sasa_embed.embedding_dim,
                           LORA_R)
            
        for i in range(len(model.encoder.function_embed)):
            model.encoder.function_embed[i] = \
                lora.Embedding(model.encoder.function_embed[i].num_embeddings, 
                               model.encoder.function_embed[i].embedding_dim)
                
    if LORA_SEQUENCE_HEAD:
        model.output_heads.sequence_head[0] = \
            lora.Linear(model.output_heads.sequence_head[0].in_features, 
                        model.output_heads.sequence_head[0].out_features, 
                        LORA_R)  
            
        model.output_heads.sequence_head[3] = \
            lora.Linear(model.output_heads.sequence_head[3].in_features, 
                        model.output_heads.sequence_head[3].out_features, 
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

def load_structure_encoder(weights_path=WEIGHTS_PATH,
                           weights_file=ENCODER_WEIGHTS_FILE_NAME):
    
    model = StructureTokenEncoder(d_model=ENCODER_D_MODEL, 
                                  n_heads=ENCODER_N_HEADS, 
                                  v_heads=ENCODER_V_HEADS, 
                                  n_layers=ENCODER_N_LAYERS, 
                                  d_out=ENCODER_D_OUT, 
                                  n_codes=ENCODER_N_CODES).eval()
        
    state_dict = torch.load("%s/%s" % (weights_path, weights_file))
    model.load_state_dict(state_dict, strict=False)
    
    return model

def load_structure_decoder(weights_path=WEIGHTS_PATH,
                           weights_file=DECODER_WEIGHTS_FILE_NAME):
    
    
    model = StructureTokenDecoder(d_model=DECODER_D_MODEL, 
                                  n_heads=DECODER_N_HEADS, 
                                  n_layers=DECODER_N_LAYERS).eval()
        
    state_dict = torch.load("%s/%s" % (weights_path, weights_file))
    model.load_state_dict(state_dict, strict=False)
    
    return model    

def load_model(lora_weights=True, 
               weights_path=WEIGHTS_PATH,
               weights_file=MODEL_WEIGHTS_FILE_NAME,
               lora_weights_file=LORA_WEIGHTS_FIlE_NAME,
               return_tokenizers=False):
    
    tokenizers = get_model_tokenizers()
    model = ESM3(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            v_heads=V_HEADS,
            n_layers=N_LAYERS,
            structure_encoder_fn=None,#ESM3_structure_encoder_v0,
            structure_decoder_fn=None,#ESM3_structure_decoder_v0,
            function_decoder_fn=None,#ESM3_function_decoder_v0,
            tokenizers=tokenizers,#get_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
                
    if lora_weights:
        model = assign_lora_weights(model)
    
    state_dict = torch.load("%s/%s" % (weights_path, weights_file))
    model.load_state_dict(state_dict, strict=False)
    
    if lora_weights:
        lora_state_dict = torch.load("%s/%s" % (weights_path, lora_weights_file))
        model.load_state_dict(lora_state_dict, strict=False)
    
    if return_tokenizers:
        return (model, tokenizers)
    
    return (model)

def load_esm2_model_and_alphabet(model_name):
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
        
    if model_name not in supported_esm2_models:
        print("Unsupported model %s" % model_name)
        
    model_weights_and_data_path = "%s/esm2/%s.pth" % (WEIGHTS_PATH, model_name)
    
    if model_weights_and_data_path in os.listdir("%s/esm2" % WEIGHTS_PATH):
        model_data = torch.load(model_weights_and_data_path)
    else:    
        model_data, regression_data = esm2.pretrained._download_model_and_regression_data(model_name)
        
        if regression_data is not None:
            model_data["model"].update(regression_data["model"])
            
        # Save model data
        torch.save(model_data, model_weights_and_data_path)
        
    return esm2.pretrained.load_model_and_alphabet_core(model_name, 
                                                        model_data, 
                                                        regression_data=None)
        

def get_wt_sequence(sequence_dataframe):
    sequence = sequence_dataframe.seq.to_list()
    mtl = sequence_dataframe.mutant.to_list()
    
    wt_seq = []
    for s,m in zip(sequence,mtl):
        position = int(m[1:(len(m) - 1)])
        original = m[0]
        wt_seq.append(s[0:position] + original + s[(position + 1):])
    
    # validation    
    all(wt_seq)
    sequence = wt_seq[0]
    
    return(sequence)
 
# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_function_dms(sequence_df):
    mutated_pos_list = sequence_df["mutant"].to_list()
    
    return([mt[0] for mt in mutated_pos_list],
           [mt[(len(mt)-1)] for mt in mutated_pos_list],
           [int(mt[1:(len(mt)-1)]) for mt in mutated_pos_list])

# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_function_gfp(sequence_df):
    
    pos = [int(x[1:]) for x in sequence_df.columns[42:64].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[42:64].to_list()]
    to_mut = [list(x) for x in sequence_df.sequence.to_list()]
    
    return [from_mut], to_mut, [pos]

    
@torch.no_grad()
def get_pssm_by_mask(esm_model, 
                     esm_tokenizer, 
                     sequence, 
                     masked_tensor,
                     esm_structure_encoder=None,
                     pdb=None,
                     make_unique=True,
                     one_forward=True,
                     use_structure=False,
                     mini_batch_size=30):

    B,S = masked_tensor.shape
    
    if make_unique:    
        masked_tensor = torch.unique(masked_tensor, dim=0)
    
    encoded_sequence = esm_tokenizer.sequence.encode(sequence)
    sequence_tokens = torch.tensor(encoded_sequence, dtype=torch.int64).reshape((1,-1))
    
    
    pad = torch.zeros(masked_tensor.shape[0]).reshape((-1,1))
    padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1).to(torch.int64)
    
    #masked_token = torch.tensor(esm_tokenizer.sequence.encode("<mask>"))[1]
    masked_token= C.SEQUENCE_MASK_TOKEN
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * sequence_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * masked_token # add masks
    
    if pdb is not None:
        
        coords, plddt, residue_index = pdb.to_structure_encoder_inputs()
        coords = coords.cpu()
        plddt = plddt.cpu()
        residue_index = residue_index.cpu()
            
        _, structure_tokens = esm_structure_encoder.encode(coords, residue_index=residue_index)
            
        structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
        structure_tokens[:, 0] = C.VQVAE_SPECIAL_TOKENS["BOS"]
        structure_tokens[:, -1] = C.VQVAE_SPECIAL_TOKENS["EOS"]       
        
        # align and add unknowns structure tokens
        if len(sequence) != len(pdb.sequence):
            
            alm = pairwise2.align.localds(pdb.sequence, 
                                          sequence, 
                                          blosum62,
                                          -20, 
                                          -1)            
            
            # copy_indices = []
            # shift_counter = 0
            # copied_counter = 0
            
            # if len(sequence) > len(pdb.sequence):
            #     misaligned_seq = alm[0].seqA
            # else:
            #     misaligned_seq = alm[0].seqB
            
            # for i,s in enumerate(misaligned_seq):
            #     if s == "-":
            #         shift_counter += 1
            #     else:
            #         copy_indices.append(copied_counter + shift_counter + 1)
            #         copied_counter += 1
            
            if len(sequence) > len(pdb.sequence):
                misaligned_seq = alm[0].seqA
            else:
                misaligned_seq = alm[0].seqB
                
            copy_indices, shift_counter, copied_counter = align_sequences(misaligned_seq)
            
                
                
            print("Got misaligned sequneces, fixing: ")
            print("\t Sequence length %d, structure length %d" % (len(sequence), len(pdb.sequence)))    
            print("\t Alignment: %s" % misaligned_seq)
            print("\t Total number of shifts %d" % shift_counter)
            
            misaligned_indices = torch.tensor([i+1 for i,c in enumerate(misaligned_seq) if c == "-"])
            
            if len(sequence) > len(pdb.sequence):
            
                aligned_structure_tokens = torch.zeros(sequence_tokens.shape, dtype=torch.int64)
                aligned_structure_tokens[:,misaligned_indices] = C.VQVAE_SPECIAL_TOKENS["PAD"]
                aligned_structure_tokens[:, 0] = C.VQVAE_SPECIAL_TOKENS["BOS"]
                aligned_structure_tokens[:, -1] = C.VQVAE_SPECIAL_TOKENS["EOS"] 
                #validate 
                print(torch.sum(aligned_structure_tokens[:,torch.tensor(copy_indices)]) == 0)
                aligned_structure_tokens[:,torch.tensor(copy_indices)] = structure_tokens[:,1:-1]            
                structure_tokens = aligned_structure_tokens
            else:
                # aligned_batched_with_mask_sequence_tokens = torch.zeros((B, structure_tokens.shape[1]), dtype=torch.int64)
                # aligned_batched_with_mask_sequence_tokens[:,misaligned_indices] = C.SEQUENCE_PAD_TOKEN
                # aligned_batched_with_mask_sequence_tokens[:, 0] = C.SEQUENCE_BOS_TOKEN
                # aligned_batched_with_mask_sequence_tokens[:, -1] = C.SEQUENCE_EOS_TOKEN
                # print(torch.sum(aligned_batched_with_mask_sequence_tokens[:,torch.tensor(copy_indices)]) == 0)
                
                # aligned_batched_with_mask_sequence_tokens[:,torch.tensor(copy_indices)] = batched_sequences_to_run_with_masks[:,1:-1]
                
                # batched_sequences_to_run_with_masks = aligned_batched_with_mask_sequence_tokens
                # # fix padded mask tensor
                # padded_masked_tensor = (batched_sequences_to_run_with_masks == masked_token).to(torch.int64)
                
                structure_tokens = structure_tokens[:,torch.tensor([0] + copy_indices + [-1])]
                   
        structure_masked_token = C.VQVAE_SPECIAL_TOKENS["MASK"]
        batched_structures_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * structure_tokens)
        batched_structures_to_run_with_masks += padded_masked_tensor * structure_masked_token # add masks                
        
        #all(sum((batched_structures_to_run_with_masks == C.VQVAE_SPECIAL_TOKENS["MASK"]).to(torch.int64) == padded_masked_tensor))
    
    
    n_batches = batched_sequences_to_run_with_masks.shape[0] // mini_batch_size    
    batched_pssm = torch.tensor([])
    
    for batch_idx in range(n_batches + 1):
        batch_s_idx = batch_idx * mini_batch_size
        batch_e_idx = (batch_idx + 1) * mini_batch_size
        if batch_e_idx > batched_sequences_to_run_with_masks.shape[0]:
            batch_e_idx = batched_sequences_to_run_with_masks.shape[0]
            
        
        print("Running first forward to calculate esm masked pssm (%d to %d)" % (batch_s_idx, batch_e_idx))
        
        if use_structure:
            batched_first_forward = esm_model.forward(sequence_tokens=batched_sequences_to_run_with_masks[batch_s_idx:batch_e_idx],
                                                      structure_tokens=batched_structures_to_run_with_masks[batch_s_idx:batch_e_idx])            
            first_forward_structure_tokens = batched_structures_to_run_with_masks[batch_s_idx:batch_e_idx]
        else:
            batched_first_forward = esm_model.forward(sequence_tokens=batched_sequences_to_run_with_masks[batch_s_idx:batch_e_idx])
            first_forward_structure_tokens = batched_first_forward.structure_logits.argmax(dim=2)
        
        if one_forward:
            batched_pssm = torch.cat([batched_pssm, batched_first_forward.sequence_logits], dim=0)
            continue
    
            
        first_forward_function_tokens = batched_first_forward.function_logits.argmax(dim=2)
        first_forward_sasa_tokens = batched_first_forward.sasa_logits.argmax(dim=2)
        first_forward_ss8_tokens = batched_first_forward.secondary_structure_logits.argmax(dim=2)
        
        batched_second_forward = esm_model.forward(sequence_tokens=batched_sequences_to_run_with_masks[batch_s_idx:batch_e_idx],
                                                   structure_tokens=first_forward_structure_tokens,
                                                   function_tokens=first_forward_function_tokens,
                                                   sasa_tokens=first_forward_sasa_tokens,
                                                   ss8_tokens=first_forward_ss8_tokens)
        
        print("Running second forward to calculate esm masked pssm (%d to %d)" % (batch_s_idx, batch_e_idx))
    
        batched_pssm = torch.cat([batched_pssm, batched_second_forward.sequence_logits], dim=0)
    
    return(batched_pssm.softmax(dim=2))
    

@torch.no_grad()
def get_fitness_from_pssm(dataset,
                          batched_pssm_tensor,
                          batched_masked_tensor,
                          original_sequence_tokens):
    
    
    B, S, V = batched_pssm_tensor.size()
    
    predicted_fitness_across_all_masks = torch.tensor([])
    fitness_across_all_masks = torch.tensor([])    
    batched_cors = []
    seq_similarity = []
    
    for idx in range(batched_pssm_tensor.shape[0]):
        
        pssm = batched_pssm_tensor[idx,1:(batched_pssm_tensor.shape[1] - 1),:]
        token_diff = pssm.argmax(dim=1) - original_sequence_tokens[1:-1]
        diff_in_seq  = int(sum(token_diff != 0))
        
        print("Overall esm thought the sequence should have %d different positions (%.3f) - #chksum %d"  % (diff_in_seq,
                                                                                                            diff_in_seq / (S-2),                                                                                                            int(sum(token_diff))))        
        seq_similarity.append(1 - (diff_in_seq / (S-2)))        
        mask = batched_masked_tensor[idx]         
        
        # Find all variants that were mutated at the mask position for pssm score calculation
        sequences_in_mask_indices = ((dataset.one_hot_mut_info[2] == mask).to(torch.int).sum(dim=1) == mask.shape[0])
        
        # Another validation, make sure we are using the correct position
        mutation_position = [int(m[1:(len(m) -1)]) for m in dataset.sequence_dataframe[sequences_in_mask_indices.detach().numpy()]["mutant"].to_list()]
        all([(mp - 1) == torch.where(mask == 1)[0] for mp in mutation_position])
        
        # ToDo: Old code that validates we were working on the correct mutations
        # old_mutation = [m[0] for m in dataset.sequence_dataframe[sequences_in_mask_indices.detach().numpy()]["mutant"].to_list()]
        # old_mutation = [esm2_alphabet.encode(aa)[0] for aa in old_mutation]
        
        # new_mutation = [m[len(m) - 1] for m in dataset.sequence_dataframe[sequences_in_mask_indices.detach().numpy()]["mutant"].to_list()]
        # new_mutation = [esm2_alphabet.encode(aa)[0] for aa in new_mutation]
                
        
        from_mutation = dataset.one_hot_mut_info[0][sequences_in_mask_indices]
        to_mutation = dataset.one_hot_mut_info[1][sequences_in_mask_indices]
        mutated_positions = mask == 1
        
        
        predicted_fitness = fitness_from_prob(pssm[mutated_positions],
                                              from_mutation[:,mutated_positions][0],
                                              to_mutation[:,mutated_positions])
        
        fitness = dataset.sequence_dataframe[sequences_in_mask_indices.detach().numpy()].log_fitness.to_list()              
        
        predicted_fitness_across_all_masks = \
            torch.cat([predicted_fitness_across_all_masks, predicted_fitness], dim=0)
        fitness_across_all_masks = \
            torch.cat([fitness_across_all_masks, torch.tensor(fitness)], dim=0)
            
        cor =  (scipy.stats.spearmanr(predicted_fitness, fitness).statistic)
        batched_cors.append(cor)
        cum_cor = scipy.stats.spearmanr(fitness_across_all_masks, predicted_fitness_across_all_masks).statistic
        
        print("Batch cor %.3f, Cum cor %.3f" % (cor, cum_cor))
        
    return (seq_similarity, 
            batched_cors, 
            scipy.stats.spearmanr(fitness_across_all_masks, 
                                  predicted_fitness_across_all_masks).statistic,                
            fitness_across_all_masks, 
            predicted_fitness_across_all_masks)
                                
    
@torch.no_grad()
def get_esm_baseline(dataset, model, tokenizers, encoder=None, pdb=None, use_structure=False):
    #sequence = dataset.sequence_dataframe.seq[0]
    sequence = get_wt_sequence(dataset.sequence_dataframe)


    masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    #masked_tensor = masked_tensor[,:]
    
    batched_pssms = get_pssm_by_mask(model, 
                                     tokenizers, 
                                     sequence, 
                                     masked_tensor, 
                                     encoder, 
                                     pdb, 
                                     make_unique=False,
                                     one_forward=True,
                                     use_structure=use_structure)
    
    return get_fitness_from_pssm(dataset,
                                 batched_pssms,
                                 masked_tensor,
                                 torch.tensor(tokenizers.sequence.encode(sequence), 
                                              dtype=torch.int64))
    

def get_esm3_baselines(use_structure=True,
                       one_forward=True,
                       override=False,
                       prefix="pretraining_baseline"):        
    
    model, tokenizers =  load_model(return_tokenizers=True)
    structure_decoder = load_structure_decoder()
    structure_encoder = None
    
    if use_structure:
        structure_encoder = load_structure_encoder()
    
        
    output_dir = "%s/zero_shot_predictions_dms/esm3" % RESULTS_PATH

    conf_prefix = "%s%s" % ("structure_" if use_structure else "",
                            "1f_" if one_forward else "2f_")
    
    output_file_name = "%s/%s_%sfitness_prediction.csv" % (output_dir, prefix, conf_prefix)
    full_values_output_file_name = "%s/%s_%sfitness_prediction_full_values.csv" % (output_dir, prefix, conf_prefix)
    
    full_path_files_in_outdir = ["%s/%s" % (output_dir, x) for x  in os.listdir(output_dir)]
    
    if output_file_name in full_path_files_in_outdir and not override:
        print("Found fitness results csvs exist! (%s)" % output_dir)
        res_df = pd.read_csv(output_file_name)
        full_values_res_df = pd.read_csv(full_values_output_file_name)
        processed_dms_datasets = pd.unique(res_df["DMS"])
    else:
        print("Couldnt find fitness results csv (%s) creating new ones" % output_dir)
        res_df = pd.DataFrame([])
        full_values_res_df = pd.DataFrame([])
        processed_dms_datasets = []
    
    
    
    dms_datasets_to_process = [dataset for dataset in all_dms_datasets if dataset not in processed_dms_datasets]

    print("Already calculated for %d datasets (%s)" % (len(processed_dms_datasets),
                                                        " ".join(processed_dms_datasets)))
    print("%d datasets left to process " % len(dms_datasets_to_process))
    
    for idx, dms_dataset_name in enumerate(dms_datasets_to_process):
        print("(%d): Analysing %s" % (idx + len(processed_dms_datasets), dms_dataset_name))
        example_dataset_path = "%s/%s/data.csv" % (BASE_DMS_PATH, dms_dataset_name)
    
        dataset = EsmSequenceActivityDataset(dataset_path=example_dataset_path,
                                             frozen_esm_model=model,#.clone().detach(),
                                             frozen_esm_tokenizer=tokenizers,
                                             frozen_esm_structure_decoder=structure_decoder,
                                             get_mutated_position_function=get_mutated_position_function_dms,
                                             pdb_class=ProteinChain,
                                             activity_column_name="log_fitness")
    
        pdb=None
        
        if use_structure:
            pdb_folder_path ="_".join(dms_dataset_name.split("_")[:2])        
            full_pdb_folder_path = "%s/%s" % (BASE_DMS_PDB_PATH, pdb_folder_path)
            
            pdb_folder_files = os.listdir(full_pdb_folder_path)
            pdb_file = [x for x in [x for  x in pdb_folder_files if x.endswith(".pdb")] 
                            if not x.endswith("scap.pdb") and not x.endswith("fix.pdb")][0]
    
            full_pdb_file_path = "%s/%s" % (full_pdb_folder_path, pdb_file)
            
            pdb = ProteinChain.from_pdb(full_pdb_file_path)
                
    
        baseline_results = get_esm_baseline(dataset,
                                            model,
                                            tokenizers,
                                            structure_encoder,
                                            pdb,
                                            use_structure)
        
        dms_res_df = pd.DataFrame({"DMS": [dms_dataset_name] * len(baseline_results[0]),
                                   "SeqSimilarity": baseline_results[0],
                                   "BatchCor": baseline_results[1],
                                   "CumCor": [baseline_results[2]] * len(baseline_results[0])})
        
        full_res_df = pd.DataFrame({"DMS": [dms_dataset_name] * len(baseline_results[3]),
                                   "Fitness": baseline_results[3],
                                   "Predicted_Fitness": baseline_results[4]})
        
        res_df = pd.concat([res_df, dms_res_df])
        full_values_res_df = pd.concat([full_values_res_df, full_res_df])
        
        print(res_df.shape)
        
        res_df.to_csv(output_file_name)
        full_values_res_df.to_csv(full_values_output_file_name)
        

@torch.no_grad()
def get_esm2_pssm_by_mask(esm2_model,
                          esm2_model_sequence_tokenizer,
                          sequence,
                          masked_tensor,
                          mini_batch_size=30):
    
    encoded_sequence = esm2_model_sequence_tokenizer.encode("<cls>" + sequence + "<eos>")
    sequence_tokens = torch.tensor(encoded_sequence, dtype=torch.int64).reshape((1,-1))
    
    pad = torch.zeros(masked_tensor.shape[0]).reshape((-1,1))
    padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1).to(torch.int64)
    
    masked_token = torch.tensor(esm2_model_sequence_tokenizer.encode("<pad>"))
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * sequence_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * masked_token # add masks
    
    n_batches = batched_sequences_to_run_with_masks.shape[0] // mini_batch_size    
    batched_pssm = torch.tensor([])
        
    for batch_idx in range(n_batches + 1):
        batch_s_idx = batch_idx * mini_batch_size
        batch_e_idx = (batch_idx + 1) * mini_batch_size
        
        if batch_e_idx > batched_sequences_to_run_with_masks.shape[0]:
            batch_e_idx = batched_sequences_to_run_with_masks.shape[0]
            
        
        print("Running masked sequence to calculate esm2 pssm (%d to %d)" % (batch_s_idx, batch_e_idx))        
        batched_esm2_logits = esm2_model.forward(batched_sequences_to_run_with_masks[batch_s_idx:batch_e_idx,:])
        
        batched_pssm = torch.cat([batched_pssm, batched_esm2_logits["logits"]], dim=0)
    
    return(batched_pssm.softmax(dim=2))    
        

@torch.no_grad()
def get_esm2_baseline(model_name, 
                      override=False,
                      prefix="pretraining_baseline"):        
    
    esm2_model,esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    
    output_dir = "%s/zero_shot_predictions_dms/esm2/%s" % (RESULTS_PATH, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_file_name = "%s/%s_fitness_prediction.csv" % (output_dir, prefix)
    full_values_output_file_name = "%s/%s_fitness_prediction_full_values.csv"  % (output_dir, prefix)
    
    full_path_files_in_outdir = ["%s/%s" % (output_dir, x) for x  in os.listdir(output_dir)]
    
    if output_file_name in full_path_files_in_outdir and not override:
        print("Found fitness results csvs exist! (%s)" % output_dir)
        res_df = pd.read_csv(output_file_name)
        full_values_res_df = pd.read_csv(full_values_output_file_name)
        processed_dms_datasets = pd.unique(res_df["DMS"])
    else:
        print("Couldnt find fitness results csv (%s) creating new ones" % output_dir)
        res_df = pd.DataFrame([])
        full_values_res_df = pd.DataFrame([])
        processed_dms_datasets = []
    
    dms_datasets_to_process = [dataset for dataset in all_dms_datasets if dataset not in processed_dms_datasets]

    print("Already calculated for %d datasets (%s)" % (len(processed_dms_datasets),
                                                        " ".join(processed_dms_datasets)))
    print("%d datasets left to process " % len(dms_datasets_to_process))
    
    for idx, dms_dataset_name in enumerate(dms_datasets_to_process):
        print("(%d): Analysing %s" % (idx + len(processed_dms_datasets), dms_dataset_name))
        example_dataset_path = "%s/%s/data.csv" % (BASE_DMS_PATH, dms_dataset_name)            
        
        dataset = Esm2SequenceActivityDataset(example_dataset_path,
                                              esm_alphabet=esm2_alphabet,
                                              get_mutated_position_function=get_mutated_position_function_dms,
                                              activity_column_name='log_fitness')
        
        
        wt_sequence = get_wt_sequence(dataset.sequence_dataframe)
        wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + wt_sequence + "<eos>"), dtype=torch.int64)
        
        masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
        
        batched_pssms =\
            get_esm2_pssm_by_mask(esm2_model, 
                                  esm2_alphabet, 
                                  wt_sequence, 
                                  masked_tensor,
                                  mini_batch_size=35)
                
        baseline_results =\
            get_fitness_from_pssm(dataset,
                                  batched_pssms,
                                  masked_tensor,
                                  wt_tokens)
            
        dms_res_df = pd.DataFrame({"DMS": [dms_dataset_name] * len(baseline_results[0]),
                                   "SeqSimilarity": baseline_results[0],
                                   "BatchCor": baseline_results[1],
                                   "CumCor": [baseline_results[2]] * len(baseline_results[0])})
        
        full_res_df = pd.DataFrame({"DMS": [dms_dataset_name] * len(baseline_results[3]),
                                   "Fitness": baseline_results[3],
                                   "Predicted_Fitness": baseline_results[4]})            
        
        res_df = pd.concat([res_df, dms_res_df])
        full_values_res_df = pd.concat([full_values_res_df, full_res_df])
        
        print(res_df.shape)
        
        res_df.to_csv(output_file_name)
        full_values_res_df.to_csv(full_values_output_file_name)
            
      
def get_esm3_gfp_baseline():
    
    base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/"
    #dataset_path = "%s/fixed_unique_gfp_sequence_dataset.csv" % base_path
    dataset_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path    
    output_dir = "%s/gfp_dataset/esm3" % RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    ROSETTA_FILES_PATH = "%s/rosetta" % ROOT_PATH
    WT_PDB_PATH = "%s/initial_data/refined.pdb" %  ROSETTA_FILES_PATH
    
    model, tokenizers =  load_model(return_tokenizers=True)
    structure_decoder = load_structure_decoder()
    structure_encoder = load_structure_encoder()
    
    
    # This is bad, I had to fish it from this link: https://github.com/Fleishman-Lab/htFuncLib/blob/main/fluroescene_vs_the_number_of_mutations/GFP_threshold_epistasis.ipynb
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"
    
    dataset = EsmSequenceActivityDataset(dataset_path,
                                         frozen_esm_model=model,#.clone().detach(),
                                         frozen_esm_tokenizer=tokenizers,
                                         frozen_esm_structure_decoder=structure_decoder,
                                         get_mutated_position_function=get_mutated_position_function_gfp,
                                         activity_column_name='is_unsorted',
                                         sequence_column_name="FullSeq",
                                         ref_seq=jonathans_reference_sequence,
                                         labels_dtype=torch.int64)
    
    masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    pad=torch.tensor(0).view((1,-1))
    padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1)
    
    wt_tokens = torch.tensor(tokenizers.sequence.encode(jonathans_reference_sequence), dtype=torch.int64).view((1,-1))
    
    sequence_mask_token = C.SEQUENCE_MASK_TOKEN
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * sequence_mask_token # add masks                
    
    first_forward = model.forward(sequence_tokens=batched_sequences_to_run_with_masks)
    pssm = first_forward.sequence_logits.softmax(dim=2)
    
    from_mutation = dataset.one_hot_mut_info[0]
    to_mutation = dataset.one_hot_mut_info[1]
    padded_mutated_positions = padded_masked_tensor == 1
    mutated_positions = masked_tensor == 1
    
    sns.heatmap(pssm[0][:,4:29].detach().numpy())
    sns.heatmap(pssm[padded_mutated_positions,:][:,4:29].detach().numpy())
    
    fixed_pssm = pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
    
    predicted_fitness =\
        fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                  from_mutation[:,mutated_positions.view((-1))][0],
                                  to_mutation[:,mutated_positions.view((-1))])
        
    first_round_predicted_fitness = predicted_fitness
    
    
    second_forward = model.forward(sequence_tokens=batched_sequences_to_run_with_masks,
                                   structure_tokens=first_forward.structure_logits.softmax(dim=2).argmax(dim=2),
                                   function_tokens=first_forward.function_logits.softmax(dim=2).argmax(dim=2),
                                   sasa_tokens=first_forward.sasa_logits.softmax(dim=2).argmax(dim=2),
                                   ss8_tokens=first_forward.secondary_structure_logits.softmax(     dim=2).argmax(dim=2))
    
    
    second_pssm = second_forward.sequence_logits.softmax(dim=2)
    
    sns.heatmap(second_pssm[0][:,4:29].detach().numpy())
    sns.heatmap(second_pssm[padded_mutated_positions,:][:,4:29].detach().numpy())
    
    fixed_second_pssm = second_pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
    
    second_round_predicted_fitness =\
        fitness_from_prob_non_dms(fixed_second_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                  from_mutation[:,mutated_positions.view((-1))][0],
                                  to_mutation[:,mutated_positions.view((-1))])        
    
    chain = ProteinChain.from_pdb(WT_PDB_PATH)
    pdb_sequence = chain.sequence
    
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cpu()
    plddt = plddt.cpu()
    residue_index = residue_index.cpu()
        
    _, structure_tokens = structure_encoder.encode(coords, residue_index=residue_index)
    
   
    alm = pairwise2.align.localds(pdb_sequence, 
                                  jonathans_reference_sequence, 
                                  blosum62,
                                  -20, 
                                  -1)    
    

                
    copy_indices, shift_counter, copied_counter = align_sequences(alm[0].seqA)    
    
    print("Got misaligned sequneces, fixing: ")
    print("\t Sequence length %d, structure length %d" % (len(jonathans_reference_sequence), len(pdb_sequence)))    
    print("\t Alignment: %s" % alm[0].seqA)
    print("\t Total number of shifts %d" % shift_counter)
            
    # +1 because wt_tokens are padded with bos /  eos
    misaligned_indices = torch.tensor([i+1 for i,c in enumerate(alm[0].seqA) if c == "-"])
    
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = C.VQVAE_SPECIAL_TOKENS["BOS"]
    structure_tokens[:, -1] = C.VQVAE_SPECIAL_TOKENS["EOS"]       
        
    aligned_structure_tokens = torch.zeros(wt_tokens.shape, dtype=torch.int64)
    aligned_structure_tokens[:,misaligned_indices] = C.VQVAE_SPECIAL_TOKENS["PAD"]
    aligned_structure_tokens[:, 0] = C.VQVAE_SPECIAL_TOKENS["BOS"]
    aligned_structure_tokens[:, -1] = C.VQVAE_SPECIAL_TOKENS["EOS"] 
    
    #validate 
    print(torch.sum(aligned_structure_tokens[:,torch.tensor(copy_indices)]) == 0)
    print("%d == %d" % (len(copy_indices), structure_tokens.shape[1] -2))
    
    aligned_structure_tokens[:,torch.tensor(copy_indices)] = structure_tokens[:,1:-1]            
    structure_tokens = aligned_structure_tokens
    
    
    first_forward_with_structure = model.forward(sequence_tokens=batched_sequences_to_run_with_masks,
                                                 structure_tokens=structure_tokens)
    
    first_forward_with_structure_pssm = first_forward_with_structure.sequence_logits.softmax(dim=2)
    
    sns.heatmap(first_forward_with_structure_pssm[0][:,4:29].detach().numpy())
    sns.heatmap(first_forward_with_structure_pssm[padded_mutated_positions,:][:,4:29].detach().numpy())
    
    fixed_first_forward_with_structure_pssm = first_forward_with_structure_pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
    
    first_forward_with_structure_predicted_fitness =\
        fitness_from_prob_non_dms(fixed_first_forward_with_structure_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                  from_mutation[:,mutated_positions.view((-1))][0],
                                  to_mutation[:,mutated_positions.view((-1))])
        
        
    second_forward_with_structure =\
        model.forward(sequence_tokens=batched_sequences_to_run_with_masks,
                      structure_tokens=structure_tokens,
                      function_tokens=first_forward_with_structure.function_logits.softmax(dim=2).argmax(dim=2),
                      sasa_tokens=first_forward_with_structure.sasa_logits.softmax(dim=2).argmax(dim=2),
                      ss8_tokens=first_forward_with_structure.secondary_structure_logits.softmax(dim=2).argmax(dim=2))    
        
    second_forward_with_structure_pssm = second_forward_with_structure.sequence_logits.softmax(dim=2)
    
    sns.heatmap(second_forward_with_structure_pssm[0][:,4:29].detach().numpy())
    sns.heatmap(second_forward_with_structure_pssm[padded_mutated_positions,:][:,4:29].detach().numpy())
    
    fixed_second_forward_with_structure_pssm = second_forward_with_structure_pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
    
    
    second_forward_with_structure_predicted_fitness =\
        fitness_from_prob_non_dms(fixed_second_forward_with_structure_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                  from_mutation[:,mutated_positions.view((-1))][0],
                                  to_mutation[:,mutated_positions.view((-1))])
        


    all_fitness = torch.stack([first_round_predicted_fitness,
                               second_round_predicted_fitness,
                               first_forward_with_structure_predicted_fitness,
                               second_forward_with_structure_predicted_fitness], dim=0)
    
    res_df = pd.DataFrame(np.transpose(all_fitness.detach().numpy()))
    res_df.to_csv("%s/predicted_fitness.csv" % output_dir)
    
    
    
def get_esm2_gfp_baseline(model_name):
    
    
    base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/"
    #dataset_path = "%s/fixed_unique_gfp_sequence_dataset.csv" % base_path
    dataset_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path    
    output_dir = "%s/gfp_dataset/esm2" % RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
    
    esm2_model,esm2_alphabet  = load_esm2_model_and_alphabet(model_name)

    # This is bad, I had to fish it from this link: https://github.com/Fleishman-Lab/htFuncLib/blob/main/fluroescene_vs_the_number_of_mutations/GFP_threshold_epistasis.ipynb
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"
    
    dataset = Esm2SequenceActivityDataset(dataset_path,
                                          esm2_alphabet,
                                          get_mutated_position_function=get_mutated_position_function_gfp,
                                          activity_column_name='is_unsorted',
                                          sequence_column_name="FullSeq",
                                          ref_seq=jonathans_reference_sequence,
                                          labels_dtype=torch.int64)
    masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    pad=torch.tensor(0).view((1,-1))
    padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1)
    
    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64).view((1,-1))
    
    sequence_mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64)
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * sequence_mask_token # add masks                
    
    logits = esm2_model.forward(batched_sequences_to_run_with_masks)
    
    
    pssm = logits["logits"].softmax(dim=2)
    
    from_mutation = dataset.one_hot_mut_info[0]
    to_mutation = dataset.one_hot_mut_info[1]
    padded_mutated_positions = padded_masked_tensor == 1
    mutated_positions = masked_tensor == 1
    
    sns.heatmap(pssm[0][:,4:29].detach().numpy())
    sns.heatmap(pssm[padded_mutated_positions,:][:,4:29].detach().numpy())
    
    fixed_pssm = pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
    
    predicted_fitness =\
        fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                  from_mutation[:,mutated_positions.view((-1))][0],
                                  to_mutation[:,mutated_positions.view((-1))])
        
    res_df = pd.DataFrame(np.transpose(predicted_fitness.detach().numpy()))
    res_df.to_csv("%s/%s_predicted_fitness.csv" % (output_dir, model_name))
    
# train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
# optimizer = None

#get_esm3_baselines()
#get_esm3_baselines(use_structure=False)
#get_esm3_baselines(one_forward=False)
#get_esm3_baselines(use_structure=False, one_forward=False)



model_name = "esm_msa1b_t12_100M_UR50S"

esm2_model,esm2_alphabet  = load_esm2_model_and_alphabet(model_name)

get_esm3_gfp_baseline()

esm2_models_to_test = [#"esm2_t33_650M_UR50D",
                       "esm1b_t33_650M_UR50S",
                       "esm1_t34_670M_UR100",
                       "esm1v_t33_650M_UR90S_1",                       
                       "esm2_t36_3B_UR50D"]

#for model_name in esm2_models_to_test:
 #   get_esm2_gfp_baseline(model_name)
    #get_esm2_baseline(model_name)
    

#get_esm2_gfp_baseline()


train_set_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/random_100k_train.csv"
train_sequences = np.isin(dataset.sequence_dataframe.sequence, ptpt.sequence)


