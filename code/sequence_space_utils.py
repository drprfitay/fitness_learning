#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:28:01 2025

@author: itayta
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

def fitness_from_prob(pssm, wt_tensor, variant_tensor):    
    wt_one_hot = torch.nn.functional.one_hot(torch.tensor(wt_tensor), pssm.shape[1])
    variant_one_hot = torch.nn.functional.one_hot(torch.tensor(variant_tensor), pssm.shape[1])   
    # return (torch.mean(torch.log(torch.sum(pos_prob_matrix * variant_one_hot, dim=1)) - torch.log(torch.sum(pos_prob_matrix * wt_one_hot, dim=1))))
    # B for batch size, S for number of mutated positions, V for vocabulary size
    #return torch.mean((torch.log(torch.einsum("BSV,SV->BSV", variant_one_hot[:,:,:].to(torch.float), pssm)) - torch.log(torch.sum(pssm * wt_one_hot, dim=1))))
    return torch.log(pssm[:,variant_tensor]).view((-1))  - torch.log(pssm[:,wt_tensor]).view((-1))

def fitness_from_prob_non_dms(pssm, wt_tensor, variant_tensor):
    wt_one_hot = torch.nn.functional.one_hot(torch.tensor(wt_tensor), pssm.shape[1])
    variant_one_hot = torch.nn.functional.one_hot(torch.tensor(variant_tensor), pssm.shape[1])   
    
    # B for batch size, S for number of mutated positions, V for vocabulary size
    return torch.mean((torch.log(torch.einsum("BSV,SV->BS", 
                                              variant_one_hot[:,:,:].to(torch.float), 
                                              pssm)) -
                       torch.log(torch.sum(pssm * wt_one_hot, dim=1))),
                      dim=1)

def fitness_from_prob_non_dms(pssm, wt_tensor, variant_tensor, device=torch.device("cpu")):
    # wt_one_hot = torch.nn.functional.one_hot(torch.tensor(wt_tensor), pssm.shape[1]).to(device)
    # variant_one_hot = torch.nn.functional.one_hot(torch.tensor(variant_tensor), pssm.shape[1]).to(device)
    wt_one_hot = torch.nn.functional.one_hot(wt_tensor.clone().detach(), pssm.shape[1]).to(device)
    variant_one_hot = torch.nn.functional.one_hot(variant_tensor.clone().detach(), pssm.shape[1]).to(device)

    
    # B for batch size, S for number of mutated positions, V for vocabulary size
    return torch.mean((torch.log(torch.einsum("BSV,SV->BS", 
                                              variant_one_hot[:,:,:].to(torch.float), 
                                              pssm)) -
                       torch.log(torch.sum(pssm * wt_one_hot, dim=1))),
                      dim=1)

def split_train_test():
    pass
    

# Mutation data_tensor should be of shape (N, 3, S)
# where N is the overall number of sequences, 
# and S is each sequence length.
#
# In this tensors there are three (N,S) tensors stacked at the 2nd dimension (dim=1)
# The 3rd (idx=2) tensor is a one_hot encoding  describing the identity of the mutations in the sequence
# The 2nd (idx=1) tensor is a one_hot encoding where instead one's are replaced by the identity (in the vocabulary) of the new AA (mutation identity)
# The 1nd (idx=0) tensor is a one_hot encoding where instead one's are replaced by the identity (in the vocabulary) of the old AA (original identity)
# 
# Fitness tensor is shaped by (N, 1) and describes the fitness of each sequence

def dms_create_contrastive_pairs(mutation_data_tensor, fitness_tensor):
    mutation_positions = torch.where(mutation_data_tensor[:,2,:] > 0)[1]
    unique_positions = torch.unique(mutation_positions)
    
    all_pairs = torch.tensor([], dtype=torch.int)
    
    for position in unique_positions:
        sequences_mutated_in_position = torch.where(mutation_positions == position)[0]
        fitness_of_sequences_in_position = fitness_tensor[sequences_mutated_in_position]
        median_fitness = torch.median(fitness_of_sequences_in_position)
        negative_sequences = sequences_mutated_in_position[torch.where(fitness_of_sequences_in_position <= median_fitness)]
        positive_sequences = sequences_mutated_in_position[torch.where(fitness_of_sequences_in_position > median_fitness)]
        
        concat_pairs = \
        torch.cat([torch.stack([torch.stack([negative,positive]) for positive in positive_sequences]) for negative in negative_sequences])
        
        all_pairs = torch.cat([all_pairs, concat_pairs])

def dms_get_working_positions(dms_df):
    return(np.unique([int(mt[1:3]) for mt in dms_df["mutant"].to_list()]))


def align_sequences(misaligned_seq):
    copy_indices = []
    shift_counter = 0
    copied_counter = 0
            
            
    for i,s in enumerate(misaligned_seq):
        if s == "-":
            shift_counter += 1
        else:
            copy_indices.append(copied_counter + shift_counter + 1)
            copied_counter += 1
            
    return copy_indices, shift_counter, copied_counter



# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_full_mask_by_first_last_colname(sequence_df, first_col, last_col, sequence_col):
    
    si = np.where(sequence_df.columns == first_col)[0][0]
    ei = np.where(sequence_df.columns == last_col)[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    to_mut = sequence_df.iloc[:,si:ei].values.tolist()#[list(x) for x in sequence_df[sequence_col].to_list()]
    
    return [from_mut], to_mut, [pos]



# Has to return a list of from, to, pos (PDB pos - one based index!!!)å
def get_mutated_position_partial_mask_by_first_last_colname(sequence_df, first_col, last_col, sequence_col):
    
    si = np.where(sequence_df.columns == first_col)[0][0]
    ei = np.where(sequence_df.columns == last_col)[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    
    # to_mut = [[aa for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df[sequence_col].to_list()]
    # pos_var = [[pos[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df[sequence_col].to_list()]
    # from_mut_var = [[from_mut[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df[sequence_col].to_list()]
    
    used_seqs = sequence_df.iloc[:,si:ei].values.tolist()
    to_mut = [[aa for i,aa in enumerate(x) if aa != from_mut[i]] for x in used_seqs]
    pos_var = [[pos[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in used_seqs]
    from_mut_var = [[from_mut[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in used_seqs]
    
    
    return from_mut_var, to_mut, pos_var


# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_function_gfp(sequence_df):
    
    si = np.where(sequence_df.columns == 'L42')[0][0]
    ei = np.where(sequence_df.columns == 'V224')[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    to_mut = [list(x) for x in sequence_df.sequence.to_list()]
    
    return [from_mut], to_mut, [pos]



# Has to return a list of from, to, pos (PDB pos - one based index!!!)
def get_mutated_position_function_gfp_n2(sequence_df):
    
    si = np.where(sequence_df.columns == 'L42')[0][0]
    ei = np.where(sequence_df.columns == 'V224')[0][0] + 1
    pos = [int(x[1:]) for x in sequence_df.columns[si:ei].to_list()]
    from_mut = [x[0] for x in sequence_df.columns[si:ei].to_list()]
    
    to_mut = [[aa for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df.sequence.to_list()]
    pos_var = [[pos[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df.sequence.to_list()]
    from_mut_var = [[from_mut[i] for i,aa in enumerate(x) if aa != from_mut[i]] for x in sequence_df.sequence.to_list()]
    
    return from_mut_var, to_mut, pos_var


def plot_hists(active, inactive, save_path=None):
    plt.hist(active, bins=30, alpha=0.6, label='Active', color='green', density=True)
    plt.hist(inactive, bins=30, alpha=0.6, label='Inactive', color='gray', density=True)
        
    plt.title("Overlaid Histograms [Separation: %.4f]" % float(np.median(active) - np.median(inactive)))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show(block=False)
    plt.pause(8)
    #time.sleep(8)
    plt.close()