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

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import SeqIO

from random import sample
from math import ceil
from collections import OrderedDict

from plm_base import *

ROOT_PATH = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
ROOT_DMS_PATH = "%s/data/datasets/DMS/Data" % ROOT_PATH
BASE_DMS_PATH = "%s/data/" % ROOT_DMS_PATH
BASE_DMS_PDB_PATH = "%s/structure_data/" % ROOT_DMS_PATH 

plm_init(ROOT_PATH)


def pairwise_cosine(X):
    X = F.normalize(X, dim=-1)
    similarity = torch.matmul(X, X.t())     # [N, N]
    distance = 1 - similarity
    return distance

def online_mine_triplets(labels):
    
    triplets = []
    
    for i, anchor_label in enumerate(labels):        
        positive_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
        negative_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]

        for pos_idx in positive_indices:
            if pos_idx == i: continue
            for neg_idx in negative_indices:
                triplets.append((i, pos_idx.item(), neg_idx.item()))
                
    return triplets


def get_indices(sequence_df, nmuts, nmuts_column="num_of_muts", rev=False, verbose=False):
        
    indices = np.repeat(False, sequence_df.shape[0])
        
    if type(nmuts) == int:
        nmuts = [nmuts]
            
    for nm in nmuts:
        indices = indices | (sequence_df[nmuts_column] == nm).to_numpy()
        if verbose:
            print("Indices included: %d" % sum(indices))
            
    if rev:
        indices = ~indices
            
    return(np.where(indices)[0].tolist())


def get_one_hot_encoding(sequence_df, first_col, last_col):
    si = np.where(sequence_df.columns == first_col)[0][0]
    ei = np.where(sequence_df.columns == last_col)[0][0]
    
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:ei]]).to_numpy()).to(torch.int64)

    
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
                 dtype=torch.double):
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
            

class SeqMLP(torch.nn.Module):
    def __init__(self, 
                 encoding_type,
                 encoding_size,
                 encoding_func,
                 plm_name=None,                         
                 hidden_layers=[1024],
                 activation="sigmoid",
                 opmode="mean",
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 tok_dropout=True,
                 device=torch.device("cpu"),                 
                 dtype=torch.double):
        super().__init__()
                 
        possible_encodings = ["onehot", "plm_embedding"]
         
        if encoding_type not in possible_encodings:
            raise Exception("Unable to support opmode %s for trunk model, allowed opmodes are: %s" % (opmode, ", ".join(possible_opmodes)))
        
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
                return self.epinnet_trunk(emb)
            
            d_in = plm_d_model * self.encoding_size  # should be num of working positions * d_model
           
        elif encoding_type == "onehot":
            self.encoding_func = encoding_func # Should return one hot encoding
            
            def encode(self, seq):
                return self.encoding_fun(seq)
            
            
            def forward(x):                
                return self.epinnet_trunk(emb)
            
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
        


def select_design_pos(seq):
    return([seq[21], seq[23], seq[24]])

epinnet = SeqMLP("plm_embedding", 3, select_design_pos, plm_name="esm2_t12_35M_UR50D")

epinnet.encode("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN")

         
# POSITION IS IN PDB (1-based idx!!!!!!)
class plmTrunkModel(torch.nn.Module):    
    def __init__(self, 
                 plm_name,
                 hidden_layers=[1024],
                 activation="sigmoid",
                 opmode="mean",
                 layer_norm=True,
                 use_bias=True,
                 activation_on_last_layer=False,
                 tok_dropout=True,
                 specific_pos=None,
                 kernel_size=20,
                 stride=5,
                 device=torch.device("cpu"),                 
                 dtype=torch.double):
        super().__init__()
        
        
        # plm = load_model(plm_name)
        # #plm, plm_tokenizer = load_esm2_model_and_alphabet(plm_name)
        # V, plm_d_model = plm.embed_tokens.weight.size()
        
        plm_obj = load_model(plm_name)
        plm = plm_obj.get_model()
        plm_tokenizer = plm_obj.get_tokenizer()
        vocab, plm_d_model = plm_obj.get_token_vocab_dim()
        V = len(vocab)
        
        self.tokenizer = plm_tokenizer
        self.plm = plm.to(device)
        self.last_layer = plm_obj.get_n_layers()
        
        # if (type(plm) == esm2.model.esm2.ESM2):
        #     self.last_layer = plm_obj.get_n_layers()
            
        def plm_forward_presentation(x):
            forward = self.plm.forward(x, repr_layers=[self.last_layer])
            hh = forward["representations"][self.last_layer]
            return(hh)
            
        self.forward_func = plm_forward_presentation                
        self.opmode = opmode
        
        possible_opmodes = ["mean", "class", "avgpool", "pos"]
        
        if opmode not in possible_opmodes:
            raise Exception("Unable to support opmode %s for trunk model, allowed opmodes are: %s" % (opmode, ", ".join(possible_opmodes)))
                        
        if opmode == "mean":            
            if specific_pos is not None:
                # Average across specific positions
                self.specific_pos = torch.tensor(specific_pos, dtype=torch.int64) - 1 # PDB INDEX!!!!!! (1-based)
                
                def emb_pool_func(hh):                
                    return(hh[:,self.specific_pos,:].mean(dim=1))
            else:
                def emb_pool_func(hh):                
                    return(hh.mean(dim=1))
            
        elif opmode == "class":
            class_token = torch.tensor(self.tokenizer.encode("<unk>"), dtype=torch.int64)
            
            def emb_pool_func(hh):
                return(hh[:,0,:])
            
        elif opmode == "avgpool":
            self.conv1d = torch.nn.AvgPool1d(kernel_size=kernel_size,stride=stride)
                
            def emb_pool_func(hh):
                return(self.conv1d(einops.rearrange(hh,"B S D->B D S")).mean(dim=2))   
        
        elif opmode == "pos":
            self.specific_pos = torch.tensor(specific_pos, dtype=torch.int64) - 1 # PDB INDEX!!!!!! (1-based)
            
            def emb_pool_func(hh):
                return(hh[:,self.specific_pos,:].flatten(1,2))
            
            
        trunk_d_in_factor = 1 if opmode != "pos" else len(self.specific_pos)
            
            
        self.emb_func = emb_pool_func
        self.epinnet_trunk = EpiNNet(d_in=plm_d_model * trunk_d_in_factor,
                                     d_out=1,                 
                                     hidden_layers=hidden_layers,
                                     activation=activation,
                                     layer_norm=layer_norm,
                                     use_bias=use_bias,
                                     activation_on_last_layer=activation_on_last_layer,
                                     device=device,                 
                                     dtype=dtype).to(device)
        
    def encode(self, seq):            
        enc_seq = ""
        if self.opmode == "class":
            enc_seq = "<unk>"
                
        enc_seq = enc_seq + "<cls>" + seq + "<eos>"
                
        return self.tokenizer.encode(enc_seq)
            

    def forward(self, x):                
        hh = self.forward_func(x)
        emb = self.emb_func(hh)
            
        return emb, hh, self.epinnet_trunk(emb)
    

class EpiNNetDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 indices,
                 encoding_function,
                 encoding_identifier,
                 cache=True,
                 sequence_column_name='full_seq',                 
                 activity_column_name='inactive',   
                 labels_dtype=torch.float64):
            
            
        if not dataset_path.endswith(".csv"):
            raise BaseException("Dataset must be a .csv file received (%s)" % dataset_path)

        
        self.dataset_path = dataset_path    
        self.sequence_dataframe = pd.read_csv(dataset_path)            
        self.size = self.sequence_dataframe.shape[0] # ToDo: read from dataset        
            
        self.sequence_column_name=sequence_column_name
        self.activity_column_name=activity_column_name
        
        self.labels = torch.tensor(self.sequence_dataframe[self.activity_column_name], dtype=labels_dtype)
                    
        self.encoding_function = encoding_function
        self.cache_path = "%s_mlp_cache/" % dataset_path.split(".csv")[0]       
        self.cache = cache
        
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs("%s/misc" % self.cache_path, exist_ok=True)   
        
        tokenized_sequences_filename = "%s_encoded_sequences.pt"  % encoding_identifier
        cached_files = os.listdir("%s/misc" % self.cache_path)
        
        
        if self.cache and tokenized_sequences_filename in cached_files:
            self.encoded_tensor = torch.load("%s/misc/%s" % (self.cache_path, tokenized_sequences_filename))
        else:
            print("Tokenizing sequences in a non-DMS dataset, this may take a while")                              
            encoded_sequences = [torch.tensor(self.encoding_function(seq)) for seq in self.sequence_dataframe[self.sequence_column_name].to_list()]
            self.encoded_tensor = torch.stack(encoded_sequences, dim=0)
            
        if self.cache:        
            print("Caching \n\t(1) %s" % (tokenized_sequences_filename))
            torch.save(self.encoded_tensor, "%s/misc/%s" % \
                       (self.cache_path, tokenized_sequences_filename))
        
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
        
        indices_tensor = torch.tensor(self.indices)#torch.tensor(sample(self.indices, 100)) ##       #
        # indices_tensor = sample(indices_tensor[self.labels[indices_tensor] == 0].tolist(), torch.unique(self.labels[indices_tensor], return_counts=True)[1][1].int()
        # ) + indices_tensor[self.labels[indices_tensor] == 1].tolist()

        self.encoded_tensor = self.encoded_tensor[indices_tensor,:]            
        self.labels = self.labels[indices_tensor]
        
        self.size = self.labels.shape[0]

    def __getitem__(self,idx):
        return self.encoded_tensor[idx], self.labels[idx]
                
    def __len__(self):
        return self.size


class EpiNNetActivityTrainTest(Dataset):
        def __init__(self,
                     train_project_name,
                     evaluation_path,
                     dataset_path,
                     train_indices,
                     test_indices,
                     encoding_function,
                     encoding_identifier,       
                     cache=True,
                     lazy_load=True,
                     sequence_column_name='full_seq',
                     activity_column_name='inactive',
                     ref_seq="",
                     mini_batch_size=20,
                     positive_label=0,
                     negative_label=1,
                     device=torch.device("cpu"),
                     labels_dtype=torch.float64):
            
            self.train_indices=train_indices,
            self.test_indices=test_indices,
            self.train_project_name=train_project_name
            self.evaluation_path=evaluation_path
            self.dataset_path=dataset_path         
            
            self.encoding_function = encoding_function
            self.encoding_identifier = encoding_identifier
            self.cache=cache
            self.lazy_load=lazy_load,
            self.sequence_column_name=sequence_column_name
            self.activity_column_name=activity_column_name
            self.ref_seq=ref_seq
            self.positive_label=positive_label
            self.negative_label=negative_label
            self.labels_dtype=labels_dtype
            self.device = device
        
            if type(self.train_indices) == tuple:
                self.train_indices = self.train_indices[0]
    
            if type(self.test_indices) == tuple:
                self.test_indices = self.test_indices[0]

            if type(dataset_path) == tuple and len(dataset_path) == 2:
                self.train_dataset_path = dataset_path[0]
                self.test_dataset_path = dataset_path[1]
                
            self.train_dataset_path = self.dataset_path
            self.test_dataset_path = self.dataset_path        
            
            self.train_dataset =\
                    EpiNNetDataset(self.train_dataset_path,
                                   self.train_indices,
                                   self.encoding_function,
                                   self.encoding_identifier,
                                   self.cache,
                                   self.sequence_column_name,
                                   self.activity_column_name,
                                   self.labels_dtype)
                    
            self.size = len(self.train_dataset)
            
        def lazy_load_func(self):
                self.test_dataset =\
                    EpiNNetDataset(self.test_dataset_path,
                                   self.test_indices,
                                   self.encoding_function,
                                   self.encoding_identifier,
                                   self.cache,
                                   self.sequence_column_name,
                                   self.activity_column_name,
                                   self.labels_dtype)
                    
        def __getitem__(self,idx):
            return self.train_dataset[idx]
                
        def __len__(self):
            return self.size           



# device = torch.device("mps")
# train_indices_func=lambda sdf: get_indices(sdf, [1,2,3], nmuts_column="num_muts")
# test_indices_func=lambda sdf: get_indices(sdf, [1,2,3], nmuts_column="num_muts", rev=True)

# # model = plmTrunkModel("esm2_t12_35M_UR50D", 
# #                       opmode="pos",
# #                       specific_pos=[1],
# #                       activation="gelu").to(device)

# model = plmTrunkModel("esm2_t12_35M_UR50D", 
#                       opmode="mean",
#                       #specific_pos=pos_to_use,
#                       activation="sigmoid",
#                       layer_norm=False,
#                       activation_on_last_layer=False).to(device)

# dataset = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv"
# ref_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"


# train_test_dataset =\
#      EpiNNetActivityTrainTest(train_project_name="",
#                              evaluation_path="",
#                              dataset_path=dataset,
#                              train_indices=train_indices_func,
#                              test_indices=test_indices_func,
#                              encoding_function=model.encode,
#                              encoding_identifier="esm2_t12_35M_UR50D",       
#                              cache=True,
#                              lazy_load=True,
#                              sequence_column_name='full_seq',
#                              activity_column_name='inactive',
#                              ref_seq=ref_seq,
#                              labels_dtype=torch.float32)



# pos_to_use = [int(x[1:]) for x in train_test_dataset.train_dataset.sequence_dataframe.columns[3:25].tolist()]


    

# train_data_loader = torch.utils.data.DataLoader(train_test_dataset, 
#                                                 batch_size=32, 
#                                                 shuffle=True)
    

# iterations = 5000
# n_epochs = ceil(iterations / len(train_data_loader))
# total_steps = 0


# optimizer = torch.optim.Adam(model.parameters(),  lr=4e-6,  weight_decay=0.1)    
# l1_loss = torch.nn.L1Loss().to(device)
# l2_loss = torch.nn.MSELoss().to(device)
# criterion = torch.nn.BCELoss()
# triplet_loss = torch.nn.TripletMarginLoss(margin=4.0,  eps=1e-7)
# model.train()

# avg_loss = torch.tensor([]).to(device)

# loss_steps = 0
# for epoch in range(0, n_epochs):
#     running_loss = torch.tensor(0, dtype=torch.float).to(device)
    
#     for iter_step, batch in enumerate(train_data_loader):
#         x = batch[0].to(device)
#         y = batch[1].to(device)
        
#         trips = torch.tensor(online_mine_triplets(y))
#         if len(trips) <= 0:
#             continue     
        
#         x = x[:,torch.tensor(pos_to_use) +1 - 1] # No +1 because <bos> encoding, pos_to_use is pdb indexed!!!
        
#         optimizer.zero_grad()
#         _, hh, y_pred = model(x)
#         #forward = model.plm(x, repr_layers=[model.plm.num_layers])
#         emb = torch.nn.functional.normalize(hh, dim=1).mean(dim=1)
#         #emb = torch.nn.functional.normalize(forward["representations"][model.plm.num_layers][:,model.specific_pos + 1,:], dim=1).flatten(1,2)
        
        
#         emb_trip = emb[trips]
#         trip_loss = triplet_loss(emb_trip[:,0,:], emb_trip[:,1,:], emb_trip[:,2,:])
#         #l2 = l2_loss(y_pred, y)
#         ce_loss = criterion(y_pred.view(-1), y)
#         total_loss = trip_loss
#         running_loss += total_loss.item()
#         total_loss.backward()        
#         optimizer.step()
        
        
#         if (iter_step + 1) % 20 == 0:
#             loss_steps += 1
#             running_loss = running_loss /  20
            
#             if len(avg_loss) == 0:
#                 avg_loss = running_loss.detach().reshape(-1)
#             else:
#                 avg_loss = torch.cat([avg_loss, running_loss.detach().reshape(-1)])
                
#             running_loss = torch.tensor(0, dtype=torch.float).to(device)
#             plt.plot(range(1, loss_steps + 1), avg_loss.cpu().detach().numpy())
#             plt.show()
            
#         print("[E%d I%d] %.3f { Triplet :%.3f CE/L2 Trunk %.3f }" % (epoch, 
#                                                              iter_step, 
#                                                              total_loss,
#                                                              trip_loss,
#                                                              ce_loss))

#         # stds = emb.std(dim=0)
#         # var_loss = torch.mean(F.relu(1 - stds))
            
#         # l1 = l1_loss(y_pred.view(-1), y)
#         # l2 = l2_loss(y_pred.view(-1), y)
        
#         # bce_loss = criterion(y_pred.view(-1), y)
#         # total_loss =  bce_loss + 5*  var_loss
#         # total_loss.backward()
#         # # loss_str ="L1: %.3f, L2: %.3f, bce_loss: %.3f, var: %.3f" % (l1.item(), l2.item(), bce_loss.item(), var_loss.item())
        
#         # # for name, param in model.epinnet_trunk.named_parameters():
#         # #     if param.grad is None:
#         # #         print(param.grad)
            
#         # print("Loss (%.3f [%s]) [Epoch %d, I %d]" %\
#         #           (total_loss.item(),
#         #            loss_str,#" ".join(loss_str),
#         #            epoch, 
#         #            iter_step))
        
        
#         # triplet_loss(emb_trip[:,0,:], emb_trip[:,1,:], emb_trip[:,2,:])
                
#         # optimizer.step()
        
def train_plm_triplet_model(plm_name: str,
                            dataset_path: str,
                            ref_seq: str,
                            save_path: str,
                            train_indices_func,
                            test_indices_func=None,
                            pos_to_use=None,
                            batch_size=32,
                            iterations=20000,
                            margin=1.0,
                            lr=4e-6,
                            weight_decay=0.1,
                            encoding_identifier=None,
                            opmode="mean",
                            hidden_layers=[1024],
                            activation="sigmoid",
                            layer_norm=False,
                            activation_on_last_layer=False,
                            device=torch.device("cpu")):    
    
    
    torch.cuda.empty_cache()


    model = plmTrunkModel(plm_name=plm_name,
                          opmode=opmode,
                          specific_pos=pos_to_use,
                          hidden_layers=hidden_layers,
                          activation=activation,
                          layer_norm=layer_norm,
                          activation_on_last_layer=activation_on_last_layer,
                          device=device).to(device)

    print("Preparing dataset...")
    train_test_dataset = EpiNNetActivityTrainTest(train_project_name="triplet_training",
                                                  evaluation_path="",
                                                  dataset_path=dataset_path,
                                                  train_indices=train_indices_func,
                                                  test_indices=test_indices_func,
                                                  encoding_function=model.encode,
                                                  encoding_identifier=encoding_identifier or plm_name,
                                                  cache=True,
                                                  lazy_load=True,
                                                  sequence_column_name='full_seq',
                                                  activity_column_name='inactive',
                                                  ref_seq=ref_seq,
                                                  labels_dtype=torch.float32,
                                                  device=device)

    if pos_to_use is None:
        pos_to_use = [int(x[1:]) for x in train_test_dataset.train_dataset.sequence_dataframe.columns[3:25].tolist()]

    print(f"Using positions: {pos_to_use}")
    train_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)
    #ce_loss_fn = torch.nn.BCELoss()
    
    model.train()
    avg_loss = torch.tensor([]).to(device)
    loss_steps = 0

    for epoch in range(n_epochs):
        running_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            
            trips = torch.tensor(online_mine_triplets(y))
            if len(trips) <= 0:
                continue     
            
            x = x[:,torch.tensor(pos_to_use) +1 - 1] # No +1 because <bos> encoding, pos_to_use is pdb indexed!!!
            
            optimizer.zero_grad()
            _, hh, y_pred = model(x)
            #forward = model.plm(x, repr_layers=[model.plm.num_layers])
            emb = torch.nn.functional.normalize(hh, dim=1).mean(dim=1)
            emb = torch.nn.functional.normalize(emb, dim=1)
            #emb = torch.nn.functional.normalize(forward["representations"][model.plm.num_layers][:,model.specific_pos + 1,:], dim=1).flatten(1,2)
            
            
            emb_trip = emb[trips]
            trip_loss = triplet_loss(emb_trip[:,0,:], emb_trip[:,1,:], emb_trip[:,2,:])
            total_loss = trip_loss
            running_loss += total_loss.item()
            total_loss.backward()        
            optimizer.step()
            
            
            if (iter_step + 1) % 20 == 0:
                loss_steps += 1
                running_loss = running_loss /  20
                
                if len(avg_loss) == 0:
                    avg_loss = running_loss.detach().reshape(-1)
                else:
                    avg_loss = torch.cat([avg_loss, running_loss.detach().reshape(-1)])
                    
                running_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(range(1, loss_steps + 1), avg_loss.cpu().detach().numpy())
                plt.show()
                
            print("[E%d I%d] %.3f { Triplet :%.3f}" % (epoch, 
                                                       iter_step, 
                                                       total_loss,
                                                       trip_loss))

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model

model = train_plm_triplet_model(
    plm_name="esm2_t12_35M_UR50D",
    dataset_path="/path/to/dataset.csv",
    ref_seq="MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN",
    save_path="/tmp/trained_plm_trunk.pt",
    train_indices_func=lambda sdf: get_indices(sdf, [1,2,3], nmuts_column="num_muts"),
    test_indices_func=lambda sdf: get_indices(sdf, [1,2,3], nmuts_column="num_muts", rev=True),
    pos_to_use=None,
    device=torch.device("cuda")  # or "cuda" or "cpu"
)