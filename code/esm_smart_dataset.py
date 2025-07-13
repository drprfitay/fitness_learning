#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 16:17:38 2025

@author: itayta
"""
import torch
import pandas as pd
import os
import re
import time

from sequence_space_utils import *
from operator import itemgetter
from random import sample

import torch.nn.functional as F
from torch.utils.data import Dataset

import warnings
#warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*add_safe_globals.*")

class EsmBaseSequenceActivityDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 sequence_tokenizer_func=None,
                 get_mutated_position_function=None,
                 cache=False,
                 sequence_column_name='seq',
                 activity_column_name='fitness',
                 ref_seq="",
                 s_i=-1,
                 e_i=-1,                 
                 labels_dtype=torch.float64):
        
        
        if not dataset_path.endswith(".csv"):
            raise BaseException("Dataset must be a .csv file received (%s)" % dataset_path)

        
        self.sequence_dataframe = pd.read_csv(dataset_path)
        
        if s_i != -1 and e_i != -1:
            self.sequence_dataframe = self.sequence_dataframe.loc[s_i:e_i,:]
        
        self.size = self.sequence_dataframe.shape[0] # ToDo: read from dataset        
        
        self.sequence_column_name=sequence_column_name
        self.activity_column_name=activity_column_name
        self.ref_seq = ref_seq
    
        
        self.labels = torch.tensor(self.sequence_dataframe[self.activity_column_name], dtype=labels_dtype)
        
        self.get_mutated_position_function = None        
        self.mut_info = None
        self.one_hot_mut_info = None
        self.sequence_tokenizer_func=sequence_tokenizer_func
        self.cache = cache
        self.cache_path = None
            
        if self.cache:
            self.cache_path = "%s_cache/" % dataset_path.split(".csv")[0]            
            os.makedirs(self.cache_path, exist_ok=True) 
            os.makedirs("%s/misc" % self.cache_path, exist_ok=True) 
            
        if get_mutated_position_function is not None:
            one_hot_mut_info_file_name = None
            mut_info_file_name = None
            
            self.get_mutated_position_function = get_mutated_position_function
            
            if self.cache:
                one_hot_mut_info_file_name = "tokenizer_%s_onehot_%s.pt" % \
                    (sequence_tokenizer_func.__name__, 
                     get_mutated_position_function.__name__)
                    
                mut_info_file_name = "tokenizer_%s_mut_%s.pt" % \
                    (sequence_tokenizer_func.__name__, 
                     get_mutated_position_function.__name__)
                    
                cached_files = os.listdir("%s/misc" % self.cache_path)
                
                if one_hot_mut_info_file_name in cached_files and mut_info_file_name in cached_files:
                    
                    print("Loading from cache \n\t(1) %s\n\t(2) %s" % (one_hot_mut_info_file_name, mut_info_file_name))
                    
                    self.one_hot_mut_info = torch.load("%s/misc/%s" % \
                                                       (self.cache_path,
                                                       one_hot_mut_info_file_name))
                        
                    self.mut_info = torch.load("%s/misc/%s" % \
                                               (self.cache_path,
                                               mut_info_file_name))
                    return
            
            # Pos mut list should be one based!!! (PDB index)
            from_mut_list, to_mut_list, pos_mut_list = get_mutated_position_function(self.sequence_dataframe)

        
            if len(from_mut_list) == 1:
                from_mut_list = from_mut_list * len(to_mut_list)
                
            if len(pos_mut_list) == 1:
                pos_mut_list = pos_mut_list * len(to_mut_list)
            
            # non_dms_dataset
            is_dms = True
            if len(from_mut_list[0]) > 1:
                if len(to_mut_list[0]) <= 1:
                    raise BaseException("Error encountered when initating a dataset, from mutation list and to mutation lists should be identical in size, (%d, %d" %(len(from_mut_list[0]), len(to_mut_list[0])))

                is_dms = False            
                    
                print("Tokenizing sequences in a non-DMS dataset, this may take a while")                              
                to_mut_list = [self.sequence_tokenizer_func("".join(x))[1:-1] for x in to_mut_list]
                from_mut_list = [self.sequence_tokenizer_func("".join(x))[1:-1] for x in from_mut_list]
            else:
                print("DMS dataset recognized!")
                from_mut_list = self.sequence_tokenizer_func("".join(from_mut_list))[1:-1]
                to_mut_list = self.sequence_tokenizer_func("".join(to_mut_list))[1:-1]
                
            
            if len(self.ref_seq) != 0:
                sequence_len = len(self.ref_seq)
            else:
                sequence_len = len(self.sequence_dataframe[self.sequence_column_name][0])
                            
            if is_dms:
                
                self.mut_info = [from_mut_list,
                                 to_mut_list,
                                 pos_mut_list] 
                
                # self.mut_info = torch.stack([torch.tensor(from_mut_list),
                #                              torch.tensor(to_mut_list),
                #                              torch.tensor(pos_mut_list) - 1], dim=1)
                
                self.one_hot_mut_info = F.one_hot(self.mut_info[:,2], sequence_len)
                self.one_hot_mut_info = torch.stack([self.one_hot_mut_info * self.mut_info[:,0].reshape((-1,1)),
                                                     self.one_hot_mut_info * self.mut_info[:,1].reshape((-1,1)),
                                                     self.one_hot_mut_info],
                                                    dim=0)
            else:
                
                print ("Generating one hot encodings for non-DMS mutations, this may take a while")
                
                            
            
                # self.mut_info = torch.stack([torch.tensor(from_mut_list),
                #                               torch.tensor(to_mut_list),
                #                               torch.tensor(pos_mut_list) - 1], dim=1)
                
                            
            
                # I FORGOT AGAINNNN ]: PDB IS 1 BASED INDEX 
                
                pos_mut_list = [[i - 1 for i in p] for p in pos_mut_list]
                self.mut_info = [from_mut_list,
                                 to_mut_list,
                                 pos_mut_list]
                # self.one_hot_mut_info = F.one_hot(self.mut_info[:,2], sequence_len)
                
                # S, M, L = self.one_hot_mut_info.size()
                
                # self.one_hot_mut_info = torch.stack([(self.one_hot_mut_info * self.mut_info[:,0].reshape((S,-1,1))).sum(dim=1),
                #                                      (self.one_hot_mut_info * self.mut_info[:,1].reshape((S,-1,1))).sum(dim=1),
                #                                      self.one_hot_mut_info.sum(dim=1)],
                #                                     dim=0)
                
                
                pos_one_hot = torch.stack([(F.one_hot(torch.tensor(pos_mut_list[i], dtype=torch.int64), sequence_len).sum(dim=0)) \
                                              for i in range(len(pos_mut_list))])
                                            
                to_mut_one_hot = torch.stack([(F.one_hot(torch.tensor(pos_mut_list[i], dtype=torch.int64), sequence_len) * \
                                               torch.tensor(to_mut_list[i], dtype=torch.int64).view((-1,1))).sum(dim=0)\
                                              for i in range(len(pos_mut_list))])
                    
                from_mut_one_hot = torch.stack([(F.one_hot(torch.tensor(pos_mut_list[i], dtype=torch.int64), sequence_len) * \
                                               torch.tensor(from_mut_list[i], dtype=torch.int64).view((-1,1))).sum(dim=0)\
                                              for i in range(len(pos_mut_list))])   
                    
                self.one_hot_mut_info = torch.stack([from_mut_one_hot, 
                                                     to_mut_one_hot,
                                                     pos_one_hot], dim=0)
            
            if self.cache:        
                    
                
                print("Caching \n\t(1) %s\n\t(2) %s" % (one_hot_mut_info_file_name, mut_info_file_name))
        
                torch.save(self.one_hot_mut_info, "%s/misc/%s" % \
                           (self.cache_path, one_hot_mut_info_file_name))
                    
                torch.save(self.mut_info, "%s/misc/%s" % \
                           (self.cache_path,mut_info_file_name))
    def __len__(self):
        return (self.size)    


class EsmSequenceActivityDataset(EsmBaseSequenceActivityDataset):
    def __init__(self,
                 dataset_path, 
                 frozen_esm_model=None, 
                 frozen_esm_tokenizer=None,
                 frozen_esm_structure_encoder=None,
                 frozen_esm_structure_decoder=None,
                 get_mutated_position_function=None,
                 pdb_class=None,
                 double_forward=True, 
                 cache=True,
                 override_cache=False,
                 smart_if_possible=True,
                 use_structure=False,
                 return_full_structural_info=False,
                 sequence_column_name='seq',
                 activity_column_name='fitness',
                 ref_seq="",
                 return_keys=["sequence","structure", "plddt", "residue_index",
                              "coords", "function", "sasa", "ss8"],
                 s_i=-1,
                 e_i=-1,
                 labels_dtype=torch.float64):
        
        
        super().__init__(dataset_path, 
                         frozen_esm_tokenizer.sequence.encode,
                         get_mutated_position_function,
                         sequence_column_name,
                         activity_column_name,
                         ref_seq,
                         s_i,
                         e_i,
                         labels_dtype)
             

        self.return_full_structural_info = return_full_structural_info
        self.frozen_esm_model=frozen_esm_model
        self.frozen_esm_tokenizer=frozen_esm_tokenizer,
        self.frozen_esm_structure_encoder=frozen_esm_structure_encoder        
        self.frozen_esm_structure_decoder=frozen_esm_structure_decoder        
        self.pdb_class=pdb_class
        
        self.double_forward=double_forward
        self.cache = cache
        self.override_cache=override_cache
        self.smart_if_possible=smart_if_possible
        self.use_structure = use_structure
        
        self.return_keys = return_keys
        
        self.cache_path = None
        

        
        if not self.return_full_structural_info:
            keys_to_remove = ["plddt", "residue_index", "coords"]
            self.return_keys = [k for k in return_keys if k not in keys_to_remove]
        
        # bizarre
        if self.frozen_esm_tokenizer is not None:
            self.frozen_esm_tokenizer = self.frozen_esm_tokenizer[0]
        
        if self.override_cache:
            return            
        
        if self.cache:
            self.cache_path = "%s_cache/" % dataset_path.split(".csv")[0]            
            os.makedirs(self.cache_path, exist_ok=True)                   
            cached_tensors = os.listdir(self.cache_path)
            
            self.sequence_dataframe = \
            pd.concat([self.sequence_dataframe, 
                       pd.DataFrame({'cache_index': [-1] * self.size})], 
                      axis=1)
            
            # ToDo: implement efficent saving if all are cached
            if len(cached_tensors) == self.size:
                pass
            else:
                self.cache_idx = 0                
                self.cache_dict = {}                

                for cached_file in cached_tensors:
                    if not bool(re.match(r"^tokens_.*\.pth$", cached_file)):
                        continue
                    
                    seq = cached_file[len("tokens_"):(len(cached_file) - len(".pth"))]
                    sequence_df_idx = self.sequence_dataframe[self.sequence_column_name] == seq
                    
                    saved_tensor = torch.load("%s/%s" % (self.cache_path, cached_file))
                    self.__add_to_cache__(sequence_df_idx, saved_tensor)
                    
                                                            
    def __add_to_cache__(self, bool_idx, tensor_dict):
                    # in case its the first cached tensor
                    if  self.cache_idx == 0:                        
                        for k,v in tensor_dict.items():
                            self.cache_dict[k] = v.unsqueeze(dim=0)
                    else:
                        for k,v in tensor_dict.items():
                            self.cache_dict[k] = torch.cat([self.cache_dict[k], v.unsqueeze(dim=0)],
                                                           dim=0)                            
                    self.sequence_dataframe.loc[bool_idx, "cache_index"] =  self.cache_idx
                    self.cache_idx += 1        
        
    @torch.no_grad()
    def __getitem__(self, idx):
        
        # Get sequence and label
        seq = self.sequence_dataframe[self.sequence_column_name].to_list()[idx]
        label = self.labels[idx]        
        mut = []
        
        if self.mut_info is not None:
            mut = [inner[idx] for inner in self.mut_info]
                    
        sequence_df_idx = self.sequence_dataframe[self.sequence_column_name] == seq
        
        idx_in_cache = \
        self.sequence_dataframe[sequence_df_idx]["cache_index"].to_list()[0]    
        
        # In case our data is cached
        if self.cache and not self.override_cache and idx_in_cache != -1:
            
            cache_keys = self.return_keys                
            cached = [self.cache_dict[k][idx_in_cache] for k in cache_keys]
            
            return_list = cached
            return_list += [torch.tensor(label)]
            return_list += mut
            return tuple(return_list)
            
        encoded_sequence = self.frozen_esm_tokenizer.sequence.encode(seq)
        sequence_tokens = torch.tensor(encoded_sequence, dtype=torch.int64).reshape((1,-1))
        
        if not self.use_structure:
            output = self.frozen_esm_model.forward(sequence_tokens=sequence_tokens)
        else:
            #ToDo get structure path
            structure_path = None
            
            pdb = self.pdb_class.from_pdb(structure_path)
            
            coords, plddt, residue_index = pdb.to_structure_encoder_inputs()
            coords = coords.cpu()
            plddt = plddt.cpu()
            residue_index = residue_index.cpu()
            
            _, structure_tokens = self.frozen_esm_structure_encoder.encode(coords, residue_index=residue_index)
            
            coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
            plddt = F.pad(plddt, (1, 1), value=0)
            structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
            structure_tokens[:, 0] = 4098
            structure_tokens[:, -1] = 4097            
            
            output = self.frozen_esm_model.forward(structure_coords=coords, 
                                                   per_res_plddt=plddt, 
                                                   structure_tokens=structure_tokens,
                                                   sequence_tokens=sequence_tokens)        
                
        first_forward_function_tokens = torch.argmax(output.function_logits, dim=-1)
        first_forward_sasa_tokens = torch.argmax(output.sasa_logits, dim=-1)
        first_forward_ss8_tokens = torch.argmax(output.secondary_structure_logits, dim=-1)
                
        # In case no structural information provided
        if not self.use_structure:        
            structure_tokens = torch.argmax(output.structure_logits, dim=-1)
            structure_tokens = (
                structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
                .where(sequence_tokens != 2, 4097)  # EOS
                .where(sequence_tokens != 31, 4100)  # Chainbreak
            )
            
            bb_coords = \
                self.frozen_esm_structure_decoder.decode(structure_tokens,
                                                         torch.ones_like(structure_tokens),
                                                         torch.zeros_like(structure_tokens))
                
            plddt = bb_coords["plddt"]            
            bb_pred = bb_coords["bb_pred"]
        
            pdb = self.pdb_class.from_backbone_atom_coordinates(bb_pred.detach(), sequence="X" + seq + "X")
            coords, _, residue_index = pdb.to_structure_encoder_inputs()
                            
        if not self.double_forward:
            tokens_dict = \
                    {"sequence": sequence_tokens,
                     "structure": structure_tokens,
                     "plddt": plddt,
                     "residue_index": residue_index,
                     "coords": coords,
                     "function": first_forward_function_tokens,
                     "sasa": first_forward_sasa_tokens,
                     "ss8": first_forward_ss8_tokens}            
                    
            if self.cache:                                        
                cached_tokens_file_name = "%s/tokens_%s.pth" % (self.cache_path, seq)
                torch.save(tokens_dict, cached_tokens_file_name)
                self.__add_to_cache__(sequence_df_idx, tokens_dict)
                                
            return_list = [tokens_dict[k] for k in self.return_keys]
            return_list += [torch.tensor(label)]
            return_list += mut
            return tuple(return_list)        
                
        # At this point, this means were double forwarding        
        output_double = \
            self.frozen_esm_model.forward(structure_coords=coords, 
                                          per_res_plddt=plddt, 
                                          structure_tokens=structure_tokens,
                                          sequence_tokens=sequence_tokens,
                                          function_tokens=first_forward_function_tokens,
                                          sasa_tokens=first_forward_sasa_tokens,                                             
                                          ss8_tokens=first_forward_ss8_tokens)
            
        second_forward_function_tokens = torch.argmax(output_double.function_logits, dim=-1)
        second_forward_sasa_tokens = torch.argmax(output_double.sasa_logits, dim=-1)
        second_forward_ss8_tokens = torch.argmax(output_double.secondary_structure_logits, dim=-1)
        
        # In case no structural information provided
        if not self.use_structure:        
            structure_tokens = torch.argmax(output_double.structure_logits, dim=-1)
            structure_tokens = (
                structure_tokens.where(sequence_tokens != 0, 4098)  # BOS
                .where(sequence_tokens != 2, 4097)  # EOS
                .where(sequence_tokens != 31, 4100)  # Chainbreak
            )
            
            bb_coords = \
                self.frozen_esm_structure_decoder.decode(structure_tokens,
                                                         torch.ones_like(structure_tokens),
                                                         torch.zeros_like(structure_tokens))
                
            plddt = bb_coords["plddt"]            
            bb_pred = bb_coords["bb_pred"]
        
            pdb = self.pdb_class.from_backbone_atom_coordinates(bb_pred.detach(), sequence="X" + seq + "X")
            coords, _, residue_index = pdb.to_structure_encoder_inputs()
    
        tokens_dict = \
                {"sequence": sequence_tokens,
                 "structure": structure_tokens,
                 "plddt": plddt,
                 "residue_index": residue_index,
                 "coords": coords,
                 "function": second_forward_function_tokens,
                 "sasa": second_forward_sasa_tokens,
                 "ss8": second_forward_ss8_tokens} 
        
        if self.cache:                                        
            cached_tokens_file_name = "%s/tokens_%s.pth" % (self.cache_path, seq)
            torch.save(tokens_dict, cached_tokens_file_name)
            self.__add_to_cache__(sequence_df_idx, tokens_dict)
        
        return_list = [tokens_dict[k] for k in self.return_keys]
        return_list += [torch.tensor(label)]
        return_list += mut
        return tuple(return_list)  
    

class Esm2SequenceActivityDataset(EsmBaseSequenceActivityDataset):
        def __init__(self,
                     dataset_path,    
                     indices=None,
                     esm_alphabet=None,
                     get_mutated_position_function=None,
                     cache=True,
                     model_name="esm2",
                     sequence_column_name='seq',
                     activity_column_name='fitness',
                     ref_seq="",
                     s_i=-1,
                     e_i=-1,
                     labels_dtype=torch.float64):
            
            self.esm_alphabet = esm_alphabet

            
            # For compatibility, esm3 encoder adds these tokens automatically
            def esm2_seq_encode(seq):
                return self.esm_alphabet.encode("<cls>" + seq + "<eos>")
            
            esm2_seq_encode.__name__ = "%s_alphabet" % model_name
            
            super().__init__(dataset_path, 
                             esm2_seq_encode,
                             get_mutated_position_function,
                             cache,
                             sequence_column_name,
                             activity_column_name,
                             ref_seq,
                             s_i,
                             e_i,
                             labels_dtype)
            
            
            tokenized_sequences_filename = "%s_tokenized_sequences.pt"  % esm2_seq_encode.__name__
            cached_files = os.listdir("%s/misc" % self.cache_path)
            
            if self.cache and tokenized_sequences_filename in cached_files:
                self.tokenized_sequences = torch.load("%s/misc/%s" % (self.cache_path, tokenized_sequences_filename))
            else:            
                self.tokenized_sequences = \
                    torch.stack([torch.tensor(esm2_seq_encode(s), dtype=torch.int64) for s in self.sequence_dataframe[self.sequence_column_name].to_list()], dim=0)
                    
                if self.cache:
                    torch.save(self.tokenized_sequences, "%s/misc/%s" % (self.cache_path, tokenized_sequences_filename))
                
            # subset based on indices
            if indices is not None:
                
                # Monkey patch!!!!!!
                if type(indices) == tuple:
                   indices = indices[0]
                    
                #subset based on indices            
                
                if callable(indices):
                    indices = indices(self.sequence_dataframe)
                    
                if type(indices) == list:
                    self.indices = indices
                    
                else:
                    self.indices = None
                    
                
            if indices is None:
                self.indices = [i for i in range(0, self.sequence_dataframe.shape[0])]
        
            indices_tensor = torch.tensor(self.indices)
            
            self.one_hot_mut_info = self.one_hot_mut_info[:,indices_tensor,:]
            self.tokenized_sequences = self.tokenized_sequences[indices_tensor,:]            
            self.mut_info = [itemgetter(*self.indices)(inner) for inner in self.mut_info]
            self.labels = self.labels[indices_tensor]
            
            
            self.wt_one_hot = (self.one_hot_mut_info[2].sum(dim=0) > 0).to(dtype=torch.int64)
            self.masked_tensor = torch.unique(self.one_hot_mut_info[2], dim=0)           

        def __getitem__(self, idx):
            return self.tokenized_sequences[idx]
    

class Esm2SequenceActivityContrastiveDataset(Esm2SequenceActivityDataset):
        def __init__(self,
                     dataset_path,    
                     indices=None,
                     esm_alphabet=None,
                     get_mutated_position_function=None,
                     cache=True,
                     model_name="esm2",
                     sequence_column_name='seq',
                     activity_column_name='fitness',
                     ref_seq="",
                     positive_label=0,
                     negative_label=1,
                     s_i=-1,
                     e_i=-1,
                     labels_dtype=torch.float64):
            
   
    
            super().__init__(dataset_path,
                             indices,
                             esm_alphabet,
                             get_mutated_position_function,
                             cache,
                             model_name,
                             sequence_column_name,
                             activity_column_name,
                             ref_seq,
                             s_i,
                             e_i,
                             labels_dtype)
            

            # self.wt_one_hot = (self.one_hot_mut_info[2].sum(dim=0) > 0).to(dtype=torch.int64)
            
            self.positive_label = positive_label
            self.negative_label = negative_label
            counts = torch.unique(self.labels, return_counts=True)[1]    
            n_pairs = torch.prod(counts)
            positives = torch.where(self.labels == positive_label)[0]
            negatives = torch.where(self.labels == negative_label)[0]
            
            negatives_ind = torch.tensor([negatives.tolist()] * len(positives)).view(n_pairs)
            positives_ind = torch.tensor([[p] * len(negatives) for p in positives.tolist()]).view(n_pairs)

            
            self.n_pairs = n_pairs                         
            self.all_pairs = torch.stack([positives_ind, negatives_ind], dim=1)
        
        def __len__(self):
            return (self.n_pairs)    
        
        def __getitem__(self, idx):
            return self.tokenized_sequences[self.all_pairs[idx][0], :], self.tokenized_sequences[self.all_pairs[idx][1], :]


class Esm2SequenceActivityContrastiveDatasetAdvancedMask(Esm2SequenceActivityDataset):
        def __init__(self,
                     dataset_path,     
                     indices=None,
                     esm_alphabet=None,
                     get_mutated_position_function=None,
                     cache=True,
                     model_name="esm2",
                     sequence_column_name='seq',
                     activity_column_name='fitness',
                     ref_seq="",
                     positive_label=0,
                     negative_label=1,
                     s_i=-1,
                     e_i=-1,
                     labels_dtype=torch.float64):
            
   
    
            super().__init__(dataset_path, 
                             indices,
                             esm_alphabet,
                             get_mutated_position_function,
                             cache,
                             model_name,
                             sequence_column_name,
                             activity_column_name,
                             ref_seq,
                             s_i,
                             e_i,
                             labels_dtype)
            
            
            # self.wt_one_hot = (self.one_hot_mut_info[2].sum(dim=0) > 0).to(dtype=torch.int64)
            # self.masked_tensor = torch.unique(self.one_hot_mut_info[2], dim=0)
            self.positive_label = positive_label
            self.negative_label = negative_label
            
                        
            positives = self.labels == positive_label
            negatives = self.labels == negative_label
            self.batch_size = 20
            
            
            self.pair_data = {}
            # For each possible mask comb)
            for m_idx in range(self.masked_tensor.shape[0]):
                # Get mask 
                mask = self.masked_tensor[m_idx]
                
                # Basically get the indices of all the variants that their one_hot encoding is identical to the mask
                variants_in_mask_indices = (self.one_hot_mut_info[2] == mask).sum(dim=1) == self.one_hot_mut_info[2].shape[1]
                
                positive_variants_in_mask_indices = torch.where(variants_in_mask_indices & positives)[0]
                negative_variants_in_mask_indices = torch.where(variants_in_mask_indices & negatives)[0]
                
                # print(positive_variants_in_mask_indices)
                # print(negative_variants_in_mask_indices)
                
                pair_size = (len(positive_variants_in_mask_indices), len(negative_variants_in_mask_indices))
                # enough variants that fit into mask to create a pair
                
                if (pair_size[0] < 1 or pair_size[1] < 1):
                    #print(pair_size)
                    continue
                
                
                negatives_ind = torch.tensor([negative_variants_in_mask_indices.tolist()] * len(positive_variants_in_mask_indices)).view(-1)
                positives_ind = torch.tensor([[p] * len(negative_variants_in_mask_indices) for p in positive_variants_in_mask_indices.tolist()]).view(-1)

            
                
                self.pair_data[m_idx] = (pair_size[0], pair_size[1],\
                                         torch.stack([positives_ind.to(torch.int64), negatives_ind.to(torch.int64)], dim=1))
            #    self.all_pairs = torch.cat([self.all_pairs, torch.stack([positives_ind.to(torch.int64), negatives_ind.to(torch.int64)], dim=1)])            
            
            example_sizes = [(m, p[0] * p[1]) for m, p in self.pair_data.items()]
            sorted_examples = example_sizes #sorted(example_sizes, key=lambda k: k[1], reverse=False)[5]
            #sorted_examples =  sorted(example_sizes, key=lambda k: k[1], reverse=False)[24:34]
            
            #self.all_pairs = torch.tensor([], dtype=torch.int64)
            self.all_pairs = {}
            self.used_masks_tensor = []
            self.used_masks = []
            self.pair_size = []
            self.n_pairs = 0
            
            for m,s in sorted_examples:
                pos_neg_indices_tensor = self.pair_data[m][2]
                
                #self.all_pairs = torch.cat([self.all_pairs, pos_neg_indices_tensor])
                self.all_pairs[m] = pos_neg_indices_tensor
                self.used_masks.append(m)
                self.used_masks_tensor.append(self.masked_tensor[m])
                self.pair_size.append((self.pair_data[m][0],
                                       self.pair_data[m][1]))
                
                self.n_pairs += pos_neg_indices_tensor.shape[0]
    
            
            self.used_masks_tensor = torch.stack(self.used_masks_tensor, dim=0)
           # self.n_pairs = self.all_pairs.shape[0]
            self.n_masks = len(self.used_masks)
        
        def __len__(self):
            #return (self.n_pairs)    
            return self.n_masks
        
        def __getitem__(self, idx):
            
            mask = self.used_masks_tensor[idx]
            pos_neg = self.all_pairs[self.used_masks[idx]]
            if pos_neg.shape[0] > self.batch_size:
                sampled_pos_neg = torch.tensor(sample(range(pos_neg.shape[0]), self.batch_size))
                pos_neg = pos_neg[sampled_pos_neg,:]
                
            # return self.tokenized_sequences[self.all_pairs[idx][0], :], self.tokenized_sequences[self.all_pairs[idx][1], :], self.one_hot_mut_info[2][self.all_pairs[idx][1]]
            return self.tokenized_sequences[pos_neg[:,0],:],\
                    self.tokenized_sequences[pos_neg[:,1],:],\
                        mask,\
                        pos_neg
    

class Esm2SequenceActivityTrainTest(Dataset):
        def __init__(self,
                     train_project_name,
                     evaluation_path,
                     dataset_path,
                     train_indices,
                     test_indices,
                     esm_model,                     
                     esm_alphabet=None,
                     full_mask_mut_positions=None,
                     partial_mask_mut_positions=None,
                     cache=True,
                     use_full_mask_only=False,
                     use_partial_mask_only=False,
                     lazy_load=True,
                     model_name="esm2",
                     sequence_column_name='seq',
                     activity_column_name='fitness',
                     ref_seq="",
                     adjusted_full_partial_ratio=1,
                     mini_batch_size=20,
                     positive_label=0,
                     negative_label=1,
                     s_i=-1,
                     e_i=-1,
                     labels_dtype=torch.float64):
            
            self.train_indices=train_indices,
            self.test_indices=test_indices,
            self.train_project_name=train_project_name
            self.evaluation_path=evaluation_path
            self.dataset_path=dataset_path         
            
            self.esm_model=esm_model
            self.esm_alphabet=esm_alphabet
            self.full_mask_mut_positions=full_mask_mut_positions
            self.partial_mask_mut_positions=partial_mask_mut_positions
            self.cache=cache
            self.use_full_mask_only=use_full_mask_only
            self.use_partial_mask_only=use_partial_mask_only
            self.lazy_load=lazy_load,
            self.model_name=model_name
            self.sequence_column_name=sequence_column_name
            self.activity_column_name=activity_column_name
            self.ref_seq=ref_seq
            self.adjusted_full_partial_ratio=adjusted_full_partial_ratio
            self.mini_batch_size=mini_batch_size
            self.positive_label=positive_label
            self.negative_label=negative_label
            self.s_i=s_i
            self.e_i=e_i
            self.labels_dtype=labels_dtype
        
            if type(self.train_indices) == tuple:
                self.train_indices = self.train_indices[0]
    
            if type(self.test_indices) == tuple:
                self.test_indices = self.test_indices[0]

            if type(dataset_path) == tuple and len(dataset_path) == 2:
                self.train_dataset_path = dataset_path[0]
                self.test_dataset_path = dataset_path[1]
                
            self.train_dataset_path = self.dataset_path
            self.test_dataset_path = self.dataset_path
        
            self.train_dataset_full_mask = \
                Esm2SequenceActivityContrastiveDataset(self.train_dataset_path,
                                                       self.train_indices,
                                                       esm_alphabet,
                                                       full_mask_mut_positions,
                                                       cache,
                                                       model_name,
                                                       sequence_column_name,
                                                       activity_column_name,
                                                       ref_seq,
                                                       positive_label,
                                                       negative_label,
                                                       e_i,
                                                       s_i,
                                                       labels_dtype)
    
            self.train_dataset_partial_mask = \
                    Esm2SequenceActivityContrastiveDatasetAdvancedMask(\
                                                           self.train_dataset_path,
                                                           self.train_indices,
                                                           esm_alphabet,
                                                           partial_mask_mut_positions,
                                                           cache,
                                                           model_name,
                                                           sequence_column_name,
                                                           activity_column_name,
                                                           ref_seq,
                                                           positive_label,
                                                           negative_label,
                                                           e_i,
                                                           s_i,
                                                           labels_dtype)
                    

                
                
            self.train_dataset_partial_mask.batch_size = self.mini_batch_size            
            self.fetch_dict = {}
            
            
            if self.use_full_mask_only:
                self.n_masks = 100
                self.__getitem___internal_ = self.fetch_full
            elif self.use_partial_mask_only:
                self.n_masks = len(self.train_dataset_partial_mask)
                self.__getitem___internal_ = self.fetch_partial
            else:
                self.n_partial_masks = len(self.train_dataset_partial_mask)
                self.n_masks = self.n_partial_masks *\
                                (1 + self.adjusted_full_partial_ratio)
                self.__getitem___internal_ = self.fetch_full_partial    
                            
                                                                            
            self.test_dataset_full_mask = None
            self.test_dataset_partial_mask = None
            
            if not self.lazy_load:
                self.lazy_load_func()
                
        def __getitem__(self,idx):
            return self.__getitem___internal_(idx)
                
        def __len__(self):
            return self.n_masks
            
        
        def fetch_partial(self, idx):
            return self.train_dataset_partial_mask[idx]
        
        def fetch_full_partial(self, idx):
            if idx < self.n_partial_masks:
                return (self.train_dataset_partial_mask[idx], "partial")
            else:
                return (self.fetch_full(), "full")
            
            
        def fetch_full(self, idx=None):
               sampled_pairs =\
                   torch.tensor(sample(range(self.train_dataset_full_mask.n_pairs),\
                                       self.mini_batch_size))
                   
                   
               pos = self.train_dataset_full_mask.all_pairs[sampled_pairs][:,0]
               neg = self.train_dataset_full_mask.all_pairs[sampled_pairs][:,1]
               
                   
               return (self.train_dataset_full_mask.tokenized_sequences[pos,:],
                       self.train_dataset_full_mask.tokenized_sequences[neg,:],
                       self.train_dataset_full_mask.masked_tensor,
                       self.train_dataset_full_mask.all_pairs[sampled_pairs])
                
        def lazy_load_func(self):
            
                if self.test_dataset_full_mask is not None and self.test_dataset_partial_mask is not None:
                    return
                
                self.test_dataset_full_mask = \
                    Esm2SequenceActivityDataset(self.test_dataset_path,
                                                self.test_indices,
                                                self.esm_alphabet,
                                                self.full_mask_mut_positions,
                                                self.cache,
                                                self.model_name,
                                                self.sequence_column_name,
                                                self.activity_column_name,
                                                self.ref_seq,
                                                self.e_i,
                                                self.s_i,
                                                self.labels_dtype)
                    
    
                self.test_dataset_partial_mask = \
                    Esm2SequenceActivityDataset(\
                                                self.test_dataset_path,
                                                self.test_indices,
                                                self.esm_alphabet,
                                                self.partial_mask_mut_positions,
                                                self.cache,
                                                self.model_name,
                                                self.sequence_column_name,
                                                self.activity_column_name,
                                                self.ref_seq,
                                                self.e_i,
                                                self.s_i,
                                                self.labels_dtype)        
                
                    
                all_masks = torch.cat([self.train_dataset_partial_mask.masked_tensor, 
                                       self.test_dataset_partial_mask.masked_tensor])
                
                self.all_masks = torch.unique(all_masks, dim=0)
                
        def evaluate_train_full(self, 
                                is_msa_transformer=False,
                                export_results=False):
            
            masked_tensor = self.train_dataset_full_mask.masked_tensor
            pad=torch.tensor(0).view((1,-1))
            padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1)
                
            padded_mutated_positions = (padded_masked_tensor == 1).view(-1)
            mutated_positions = masked_tensor == 1
            
            wt_tokens = torch.tensor(self.esm_alphabet.encode("<cls>" + self.ref_seq + "<eos>"), dtype=torch.int64).view((1,-1))
            eos_token = torch.tensor(self.esm_alphabet.encode("<eos>"), dtype=torch.int64)    
            mask_token = torch.tensor(self.esm_alphabet.encode("<mask>"), dtype=torch.int64)
            
            batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
            batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token
            
            from_mutation = self.train_dataset_full_mask.one_hot_mut_info[0]
            to_mutation = self.train_dataset_full_mask.one_hot_mut_info[1]
            
            positives = self.train_dataset_full_mask.labels == 0
            negatives = self.train_dataset_full_mask.labels == 1
            
            if is_msa_transformer:
                batched_sequences_to_run_with_masks = batched_sequences_to_run_with_masks.view((1,1,-1))
                logits = self.esm_model(batched_sequences_to_run_with_masks)
                masked_logits = logits["logits"][:,0,:,:]
                pssm = masked_logits.softmax(dim=2)
                fixed_pssm = pssm[:,1:-1,:].view((len(self.ref_seq), -1))
                  
                predicted_fitness =\
                      fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                      from_mutation[:,mutated_positions.view((-1))][0],
                      to_mutation[:,mutated_positions.view((-1))])
                  
                ina = predicted_fitness[positives].detach().numpy()
                act = predicted_fitness[negatives].detach().numpy()
                  
                plot_hists(act, ina)                
            
        def evaluate_full(self,
                          train=True,
                          is_msa_transformer=False,
                          return_results=False,
                          return_act_inact=False,
                          device=torch.device("cpu"),
                          plot=False,
                          save=False):
            
            self.esm_model.to(device)
            
            
            if train:                
                eval_dataset = self.train_dataset_full_mask
            else:
                eval_dataset = self.test_dataset_full_mask
                
                if eval_dataset is None:
                    self.lazy_load_func()
                    eval_dataset = self.train_dataset_full_mask
            
            masked_tensor = eval_dataset.masked_tensor
            pad=torch.tensor(0).view((1,-1))
            padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1)
                
            padded_mutated_positions = (padded_masked_tensor == 1).view(-1)
            mutated_positions = masked_tensor == 1
            
            wt_tokens = torch.tensor(self.esm_alphabet.encode("<cls>" + self.ref_seq + "<eos>"), dtype=torch.int64).view((1,-1))
            eos_token = torch.tensor(self.esm_alphabet.encode("<eos>"), dtype=torch.int64)    
            mask_token = torch.tensor(self.esm_alphabet.encode("<mask>"), dtype=torch.int64)
            
            batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
            batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token
            
            from_mutation = eval_dataset.one_hot_mut_info[0]
            to_mutation = eval_dataset.one_hot_mut_info[1]
            
            positives = eval_dataset.labels == 0
            negatives = eval_dataset.labels == 1
            
            predicted_fitness = None
            
            view_shape = (1,-1)
            softmax_dim = 1
            
            if is_msa_transformer:
                view_shape = (1,1,-1)
                softmax_dim = 2
                
            batched_sequences_to_run_with_masks = batched_sequences_to_run_with_masks.view(view_shape)
            logits = self.esm_model(batched_sequences_to_run_with_masks.to(device))
            
            if is_msa_transformer:
                masked_logits = logits["logits"][:,0,1:-1,:]
            else:
                masked_logits = logits["logits"][0,1:-1,:]
                
            pssm = masked_logits.softmax(dim=softmax_dim).view((len(self.ref_seq), -1))
            
            #fixed_pssm = pssm[:,1:-1,:].view((len(self.ref_seq), -1))
                  
            predicted_fitness =\
                fitness_from_prob_non_dms(pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                          from_mutation[:,mutated_positions.view((-1))][0],
                                          to_mutation[:,mutated_positions.view((-1))],
                                          device=device)
                  
            act = predicted_fitness[positives].cpu().detach().numpy()
            ina = predicted_fitness[negatives].cpu().detach().numpy()
                  
            
            if plot:
                if save:
                    save_path = "%s/best_sep_on_validatino.png" % self.evaluation_path
                else:
                    save_path = None
                plot_hists(act, ina, save_path=save_path)                   
        
            if return_act_inact:
                return(act, ina)
            
            return(predicted_fitness)
        
            # if return_results:            
            #     eval_df =\
            #         pd.DataFrame({"sequence" : eval_dataset.sequence_dataframe[self.sequence_column_name],
            #                       "fitness_score": predicted_fitness.detach().numpy()})
            #     return(eval_df)
    
        @torch.no_grad()
        def evaluate_across_masks(self,
                                  hp = False,
                                  override=False,
                                  cache=True,
                                  flush_cache_every=15,
                                  device=torch.device("cpu")):                 
            if self.test_dataset_full_mask is None:
                self.lazy_load_func()                
            
            save_path = "%s/%s" % (self.evaluation_path, self.train_project_name)
            os.makedirs(save_path, exist_ok=True)
            
            train_results_filename = "train_results.pt" 
            test_results_filename = "test_results.pt"
            files_in_cache = os.listdir(save_path)
                                            

            if hp:
                inf_model = self.esm_model.half()
            else:
                inf_model = self.esm_model
                
            self.esm_model.to(device)
                
                
            test_dataset = self.test_dataset_partial_mask
            train_dataset = self.train_dataset_partial_mask
            
            test_dataset.one_hot_mut_info = test_dataset.one_hot_mut_info.to(device)
            train_dataset.one_hot_mut_info = train_dataset.one_hot_mut_info.to(device)
            
            n_masks, S = self.all_masks.size()
            batch_size = 30              
        
            n_batches = n_masks // batch_size
           
            wt_tokens = torch.tensor(self.esm_alphabet.encode("<cls>" + self.ref_seq + "<eos>"), dtype=torch.int64, device=device).view((1,-1))
            eos_token = torch.tensor(self.esm_alphabet.encode("<eos>"), dtype=torch.int64, device=device)    
            mask_token = torch.tensor(self.esm_alphabet.encode("<mask>"), dtype=torch.int64, device=device)
            
            
            
            results_pulled_from_cache = False                         
            
            initial_batch_idx = 0
            
            if cache and\
               not override and\
               train_results_filename in files_in_cache and\
               test_results_filename in files_in_cache:
                
                
                train_results = torch.load("%s/%s" % (save_path, train_results_filename)).to(device)
                test_results = torch.load("%s/%s" % (save_path, test_results_filename)).to(device)
                
                initial_mask_idx = max(torch.max(train_results[:,0]), 
                                        torch.max(test_results[:,0])).item()
                
                initial_mask_idx = int(initial_mask_idx)
                
                initial_mask_idx
                initial_batch_idx = initial_mask_idx // batch_size
                initial_batch_idx += 1
                
                # Probably collapsed or didnt finish batch?
                if initial_batch_idx * batch_size - 1 != initial_mask_idx:
                    initial_batch_idx -= 1
                    
                print("Cached evaluation results found, starting from batch %d (mask: %d)" % (initial_batch_idx, initial_mask_idx))
                results_pulled_from_cache = True
                
            # In case we didn't want to override but truly didnt find results            
            self.evaluate_full(train=False, device=device, plot=True)
            
            if not results_pulled_from_cache:
                
                train_fitness_full = self.evaluate_full(train=True, device=device, plot=True)
                test_fitness_full = self.evaluate_full(train=False, device=device, plot=True)
                        
                train_results = torch.ones((len(train_dataset.indices), 5), dtype=torch.float32, device=device) * -1
                test_results = torch.ones((len(test_dataset.indices), 5), dtype=torch.float32, device=device) * -1
                
                 
                # positives_train = train_dataset.labels == 0
                # negatives_train = train_dataset.labels == 1
                # positives_test = test_dataset.labels == 0
                # negatives_test =  test_dataset.labels == 1
                
                train_results[:,4] = torch.tensor(train_dataset.indices)
                test_results[:,4] = torch.tensor(test_dataset.indices)
                train_results[:,3] = train_dataset.labels
                test_results[:,3] = test_dataset.labels
                train_results[:,2] = train_fitness_full
                test_results[:,2] = test_fitness_full
                    
                
                masks_df = pd.DataFrame(self.all_masks.detach().numpy())
                
                #torch.save(self.all_masks, "%s/evaluated_sequence_masks_onehot.csv" % save_path)
                masks_df.to_csv("%s/evaluated_sequence_masks_onehot.csv" % save_path, index=False)
                
            mask_to_indices_json = {}
            
            for batch_idx in range(initial_batch_idx, n_batches + 1):
                
                # if (batch_idx == 100):
                #     etime = time.time()
                #     print("########## %.3f" % etime - stime)
                #     raise BaseException("done")
                    
                mask =  n_masks
                s_i = batch_idx * batch_size
                e_i = (batch_idx + 1) * batch_size
               
                if e_i > n_masks: 
                    e_i = n_masks                   
                  
                masks = self.all_masks[s_i:e_i].to(device)            
                pad=torch.zeros(masks.shape[0], dtype=torch.int64, device=device).view((-1,1))
                padded_masked_tensor = torch.cat([pad, masks, pad], dim=1)
                padded_mutated_positions = (padded_masked_tensor == 1)            
                batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64, device=device) - padded_masked_tensor) * wt_tokens[0,:])
                batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks    
                
                before = time.time()
                logits = inf_model(batched_sequences_to_run_with_masks)
                after = time.time()
                print("(%d) - forward time : %.6f" % (batch_idx, after - before))
                
                for i in range(0, masks.shape[0]):
                    mask = masks[i]
                    
                    # if sum(mask) == 0: 
                    #     continue
                    
                    mutated_positions = mask == 1
                    
                    masked_logits = logits["logits"][i,1:-1,:]                                        
                    pssm = masked_logits.softmax(dim=1).view((len(self.ref_seq), -1))
                    
                    
                    test_indices = torch.where((test_dataset.one_hot_mut_info[2,] == mask).sum(dim=1) == S)[0]
                    train_indices = torch.where((train_dataset.one_hot_mut_info[2,] == mask).sum(dim=1) == S)[0]
                    
                    train_from_mutation = None
                    train_to_mutation = None
                    train_len = len(train_indices)
                    test_len = len(test_indices)
                    
      
                    if test_len > 0 and train_len > 0:
                        train_from_mutation = train_dataset.one_hot_mut_info[0][train_indices,:]
                        train_to_mutation = train_dataset.one_hot_mut_info[1][train_indices,:]
                        
                        test_from_mutation = test_dataset.one_hot_mut_info[0][test_indices,:]
                        test_to_mutation = test_dataset.one_hot_mut_info[1][test_indices,:]
                       
                        from_mutation = torch.cat([train_from_mutation,
                                                   test_from_mutation], dim=0)
                        to_mutation = torch.cat([train_from_mutation,
                                                 test_from_mutation], dim=0)
                        
                        train_s = 0
                        train_e = train_len
                        test_s = train_len
                        test_e = train_len + test_len
                        

                    elif train_len > 0:
                        train_from_mutation = train_dataset.one_hot_mut_info[0][train_indices,:]
                        train_to_mutation = train_dataset.one_hot_mut_info[1][train_indices,:]
                        
                        from_mutation = train_from_mutation
                        to_mutation = train_to_mutation
                        
                        train_s= 0
                        train_e = train_len
                        test_s = train_len
                        test_e = train_len
                        
                        
                        
                    elif test_len > 0:
                        test_from_mutation = test_dataset.one_hot_mut_info[0][test_indices,:]
                        test_to_mutation = test_dataset.one_hot_mut_info[1][test_indices,:]
                        
                        from_mutation = test_from_mutation
                        to_mutation = test_to_mutation
                        train_s = 0
                        train_e = 0
                        test_s = 0
                        test_e = test_len
                    
                    
                    #print ("\t Tr %d - Te %d" % (train_len, test_len))
                    predicted_fitness =\
                        fitness_from_prob_non_dms(pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                                  from_mutation[:,mutated_positions.view((-1))][0],
                                                  to_mutation[:,mutated_positions.view((-1))],
                                                  device=device)
                        
                    if test_len > 0:
                        test_results[test_indices,0] = s_i + i
                        test_results[test_indices,1] =  predicted_fitness[test_s:test_e]
                        
                        # mean_act = test_results[test_indices,3][test_results[test_indices,5] == 0,].mean()
                        # mean_inact = test_results[test_indices,3][test_results[test_indices,5] == 1,].mean()
                        # print("Active (%.3f), Inactive (%.3f)" % (mean_act, mean_inact))
                        
                        
                    if train_len > 0:
                        train_results[train_indices,0] = s_i + i
                        train_results[train_indices,1] =  predicted_fitness[train_s:train_e]
                        
                        
                if cache and batch_idx % flush_cache_every == 0:
                    print("Cached flushed at batch idx %d" % batch_idx)
                    torch.save(train_results.cpu(), "%s/%s" % (save_path, train_results_filename))
                    torch.save(test_results.cpu(), "%s/%s" % (save_path, test_results_filename))

            torch.save(train_results.cpu(), "%s/%s" % (save_path, train_results_filename))
            torch.save(test_results.cpu(), "%s/%s" % (save_path, test_results_filename))
            train_df = pd.DataFrame(train_results.cpu().detach().numpy())
            test_df = pd.DataFrame(test_results.cpu().detach().numpy())
            train_df.to_csv("%s/%s" % (save_path, train_results_filename.replace(".pt", ".csv")))
            test_df.to_csv("%s/%s" % (save_path, test_results_filename.replace(".pt", ".csv")))
            
                