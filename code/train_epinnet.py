#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:26:20 2025

@author: itayta
"""


import sys, os
import torch
from torch.cpu import device_count
import torch.nn.functional as F
import loralib as lora
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import einops

import yaml
import argparse
import time

from esm_smart_dataset import *
from sequence_space_utils import *

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import SeqIO
from sklearn.manifold import TSNE

from random import sample
from math import ceil
from collections import OrderedDict

from plm_base import *

import yaml

# Example config.yaml:
# ---
# root_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
# dataset_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv"
# save_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/one_shot/"
# weights_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/final_model.pt"

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

root_path = config["root_path"]
dataset_path = config["dataset_path"]
save_path = config["save_path"]
weights_path = config["weights_path"]


ROOT_PATH = root_path
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


def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:ei]]).to_numpy()).to(torch.int64)

    return(one_hot_encoding)
    
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

# epinnet = SeqMLP("plm_embedding", 3, select_design_pos, plm_name="esm2_t12_35M_UR50D")

# epinnet.encode("MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN")

     
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
                 trunk_classes=2,
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
                                     d_out=trunk_classes,                 
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
            

    def _emb_only_forward(self, x):
        return self.forward_func(x)

    def _forward(self, x):                
        hh = self.forward_func(x)
        emb = self.emb_func(hh)
            
        return emb, hh, self.epinnet_trunk(emb)
    
    def forward(self, x):
        return self._emb_only_forward(x)


class EpiNNetDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 indices,
                 encoding_function,
                 encoding_identifier,
                 external_encoding=None,
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
    
        
        if external_encoding is None:
        
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
                    
                    
        else:
            self.encoded_tensor = external_encoding
        
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
        
        indices_tensor = torch.tensor(self.indices)#torch.tensor(sample(self.indices, 100)) ##       #torch.tensor(sample(self.indices, 1000)) ##       #
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
                     external_encoding=None,
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
            self.external_encoding = external_encoding
            self.cache=cache
            self.lazy_load=lazy_load
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
                                   self.external_encoding,
                                   self.cache,
                                   self.sequence_column_name,
                                   self.activity_column_name,
                                   self.labels_dtype)
                    
            self.size = len(self.train_dataset)

            self.loaded = False

            if not self.lazy_load:
                self.loaded = True
                self.lazy_load_func()
            
        def lazy_load_func(self):

                if not self.loaded:
                    self.test_dataset =\
                        EpiNNetDataset(self.test_dataset_path,
                                    self.test_indices,
                                    self.encoding_function,
                                    self.encoding_identifier,
                                    self.external_encoding,
                                    self.cache,
                                    self.sequence_column_name,
                                    self.activity_column_name,
                                    self.labels_dtype)
                    self.loaded = True
                    
        def __getitem__(self,idx):
            return self.train_dataset[idx]
                
        def __len__(self):
            return self.size  
            
        @torch.no_grad()
        def evaluate(self,
                     model,
                     eval_func,
                     finalize_func,
                     eval_train=True,
                     eval_test=True,
                     internal_batch_size=20):

            model = model.to(self.device)

            confs = []
            if eval_train:
                confs.append({"train": True,
                              "path_prefix": "train"})

            if eval_test:
                confs.append({"train": False,
                              "path_prefix": "test",
                              "subsample": "subsample.csv"})            

            for conf in confs:
                is_train = conf["train"]
                path_prefix = conf["path_prefix"]

                if is_train:
                    print("Evaluating train dataset")
                    working_dataset = self.train_dataset
                else:
                    print("Evaluating test dataset")
                    self.lazy_load_func()
                    working_dataset = self.test_dataset

                if "subsample" in conf:
                    from torch.utils.data import Subset
                    subsample_path = conf["subsample"]
                    subsample_df = pd.read_csv(subsample_path)
                    if "indices" in subsample_df.columns:
                        print(f"Sampling dataset from size {len(working_dataset)} to size {len(subsample_df['indices'])}")
                        subsample_indices = subsample_df["indices"].tolist()
                        working_dataset = Subset(working_dataset, subsample_indices)
                    else:
                        print(f"Warning: 'indices' column not found in {subsample_path}")
                            
                dataloader =\
                     torch.utils.data.DataLoader(working_dataset, batch_size=internal_batch_size, shuffle=False)

                aggregated_evaluated_data = {}

                for idx, data in enumerate(dataloader):
                    aggregated_evaluated_data = eval_func(model, data, aggregated_evaluated_data, self.device)
                    
                    if "trip_loss" in aggregated_evaluated_data and (idx + 1) % 20 == 0:
                        print("Iter %d [%.3f]" % (idx, idx/len(dataloader)))
                        trip_loss_data = aggregated_evaluated_data["trip_loss"].cpu().numpy() if hasattr(aggregated_evaluated_data["trip_loss"], "cpu") else aggregated_evaluated_data["trip_loss"]
                        plt.figure()
                        plt.hist(trip_loss_data, bins=30)
                        plt.title(f"Triplet Loss Histogram at batch {idx+1}")
                        plt.xlabel("Triplet Loss")
                        plt.ylabel("Frequency")
                        plt.draw()
                        plt.pause(0.001)
                        plt.close()
                                    
                evaluated_data_to_save = finalize_func(aggregated_evaluated_data, working_dataset)

                prefix_folder = os.path.join(self.evaluation_path, path_prefix)
                os.makedirs(prefix_folder, exist_ok=True)

                for key, value in evaluated_data_to_save.items():
                    filename = f"{key}"
                    # Create the prefix folder within evaluation_path                    
                    # Save matplotlib figure
                    if hasattr(value, "savefig"):
                        fig_path = os.path.join(prefix_folder, f"{filename}.png")
                        value.savefig(fig_path)
                        plt.close(value)
                    # Save pandas DataFrame
                    elif hasattr(value, "to_csv"):
                        csv_path = os.path.join(prefix_folder, f"{filename}.csv")
                        value.to_csv(csv_path, index=False)
                    # Save torch tensor
                    elif hasattr(value, "cpu") and hasattr(value, "detach"):
                        tensor_path = os.path.join(prefix_folder, f"{filename}.pt")
                        torch.save(value.cpu().detach(), tensor_path)
                    # Save numpy array
                    elif hasattr(value, "shape") and hasattr(value, "dtype"):
                        npy_path = os.path.join(prefix_folder, f"{filename}.npy")
                        np.save(npy_path, value)
                    else:
                        # Optionally, skip or print warning for unknown types
                        print(f"Warning: Could not save {key}, unknown type {type(value)}")

            
def train_plm_triplet_model(
    plm_name: str,
    save_path: str,
    train_test_dataset,
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
    device=torch.device("cpu"),
    model=None  
):
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if model is None:
        model = plmTrunkModel(
            plm_name=plm_name,
            opmode=opmode,
            specific_pos=pos_to_use,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            activation_on_last_layer=activation_on_last_layer,
            device=device
        ).to(device)
    else:
        model = model.to(device)

    if train_test_dataset is None:
        raise ValueError("train_test_dataset must be provided as an argument.")

    if pos_to_use is None:
        pos_to_use = [int(x[1:]) for x in train_test_dataset.train_dataset.sequence_dataframe.columns[3:25].tolist()]

    print(f"Using positions: {pos_to_use}")
    train_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)
    
    model.train()
    avg_loss = torch.tensor([]).to(device)
    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            trips = torch.tensor(online_mine_triplets(y))

            if len(trips) <= 0:
                continue     
            optimizer.zero_grad()

            hh = model(x)

            emb = torch.nn.functional.normalize(hh[:,torch.tensor(pos_to_use),:], dim=1).mean(dim=1)
            emb = torch.nn.functional.normalize(emb, dim=1)
            emb_trip = emb[trips]

            trip_loss = triplet_loss(emb_trip[:,0,:], emb_trip[:,1,:], emb_trip[:,2,:])
            total_loss = trip_loss

            epoch_loss += total_loss.item()
            iter_20b_loss += total_loss.item()

            total_loss.backward()        
            optimizer.step()

            total_steps += 1
            running_batch_loss = torch.cat([running_batch_loss, total_loss.detach().reshape(-1)])

            if (step + 1) % 20 == 0:
                total_steps += 1
                iter_20b_loss = iter_20b_loss /  20
                running_20b_loss = torch.cat([running_20b_loss, iter_20b_loss.detach().reshape(-1)])
                iter_20b_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(range(1, running_20b_loss.shape[0] + 1), running_20b_loss.cpu().detach().numpy())
                plt.show()
            print("[E%d I%d] %.3f { Triplet :%.3f}" % (epoch, step, total_loss, trip_loss))
            if total_steps % 1000 == 0:
                print("\t\tCheckpoint [%d]" % total_steps)
                torch.save(model.state_dict(), save_path + "checkpoint_model_%d.pt" % total_steps)
                torch.save(running_batch_loss.cpu().detach(), save_path + "batch_loss.pt")
                torch.save(running_epoch_loss.cpu().detach(), save_path + "epoch_loss.pt")
                torch.save(running_20b_loss.cpu().detach(), save_path + "20b_loss.pt")
        running_epoch_loss = torch.cat([running_epoch_loss, epoch_loss.detach().reshape(-1)])
    torch.save(model.state_dict(), save_path + "final_model.pt")
    torch.save(running_batch_loss.cpu().detach(), save_path + "batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), save_path + "epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), save_path + "20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model


def train_epinnet(
    train_test_dataset,
    save_path: str,
    encodings=None,
    model=None,
    batch_size=32,
    iterations=20000,                  
    lr=1e-7,
    weight_decay=0.1,
    hidden_layers=[1024],
    activation="sigmoid",
    layer_norm=True,
    activation_on_last_layer=False,
    device=torch.device("cpu"),
):

    torch.cuda.empty_cache()

    # Initialize model if not provided
    if model is None:
        if encodings is None:
            raise ValueError("If model is not provided, encodings must be passed to initialize the model.")
        model = EpiNNet(
            encodings.shape[1],
            2,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            activation_on_last_layer=activation_on_last_layer,
            device=device
        ).to(device)

    # One-hot encode labels for training
    train_test_dataset.train_dataset.labels = torch.nn.functional.one_hot(
        train_test_dataset.train_dataset.labels.to(torch.long), 2
    ).to(torch.float)

    train_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()

    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            y_pred = model(x)

            total_loss = ce_loss_fn(y_pred, y)
            epoch_loss += total_loss.item()
            iter_20b_loss += total_loss.item()
            total_loss.backward()        
            optimizer.step()
            total_steps += 1

            running_batch_loss = torch.cat([running_batch_loss, total_loss.detach().reshape(-1)])
            
            if (step + 1) % 20 == 0:
                iter_20b_loss = iter_20b_loss / 20
                running_20b_loss = torch.cat([running_20b_loss, iter_20b_loss.detach().reshape(-1)])
                iter_20b_loss = torch.tensor(0, dtype=torch.float).to(device)
                plt.plot(range(1, total_steps + 1), running_20b_loss.cpu().detach().numpy())
                plt.show()
                
            print("[E%d I%d] %.3f" % (epoch, step, total_loss))
        running_epoch_loss = torch.cat([running_epoch_loss, epoch_loss.detach().reshape(-1)])
        
    # # Usage:
    # train_test_dataset.lazy_load_func()
    # train_test_dataset.test_dataset.labels = torch.nn.functional.one_hot(
    #     train_test_dataset.test_dataset.labels.to(torch.long), 2
    # ).to(torch.float)
    
    # eval_batch_size = 500
    # test_pred = torch.tensor([])
    # test_loader = torch.utils.data.DataLoader(
    #     train_test_dataset.test_dataset, 
    #     batch_size=eval_batch_size, 
    #     shuffle=False
    # )
    
    # test_score = torch.tensor([], dtype=torch.float)
    # with torch.no_grad():
    #     for i, batch in enumerate(test_loader):
    #         print("Evaluating test batch %d" % i)
    #         x = batch[0].to(device)
    #         y = batch[1].to(device)
    #         y_pred = model(x)
    #         test_pred = torch.cat([test_pred, y_pred.argmax(dim=1).cpu().detach()], dim=0)
    #         test_score = torch.cat([test_score, y_pred.softmax(dim=1)[:,0].cpu().detach()])
            
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path + "/model.pt")
    torch.save(running_batch_loss.cpu().detach(), save_path + "batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), save_path + "epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), save_path + "20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model

@torch.no_grad()
def epinnt_evaluate_function(model, data, aggregated_evaluated_data, device=torch.device("cpu")):
    x = data[0].to(device)
    y = data[1].to(device)
    y_pred = model(x)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    batch_loss = ce_loss_fn(y_pred, y)

    # Accumulate predictions, scores, true labels, and loss
    if "test_pred" not in aggregated_evaluated_data:
        aggregated_evaluated_data["test_pred"] = torch.tensor([], dtype=torch.long)
    if "test_score" not in aggregated_evaluated_data:
        aggregated_evaluated_data["test_score"] = torch.tensor([], dtype=torch.float)
    if "true_labels" not in aggregated_evaluated_data:
        aggregated_evaluated_data["true_labels"] = torch.tensor([], dtype=torch.long)
    if "loss" not in aggregated_evaluated_data:
        aggregated_evaluated_data["loss"] = torch.tensor([], dtype=torch.float)

    aggregated_evaluated_data["test_pred"] = torch.cat(
        [aggregated_evaluated_data["test_pred"], y_pred.argmax(dim=1).cpu().detach()], dim=0
    )
    aggregated_evaluated_data["test_score"] = torch.cat(
        [aggregated_evaluated_data["test_score"], y_pred.softmax(dim=1)[:, 0].cpu().detach()]
    )
    aggregated_evaluated_data["true_labels"] = torch.cat(
        [aggregated_evaluated_data["true_labels"], y.argmax(dim=1).cpu().detach()]
    )
    aggregated_evaluated_data["loss"] = torch.cat(
        [aggregated_evaluated_data["loss"], batch_loss.detach().reshape(-1).cpu()]
    )

    return aggregated_evaluated_data

@torch.no_grad()
def epinnt_finalize_function(aggregated_evaluated_data, dataset):
    # Restore original finalize logic
    true_labels = aggregated_evaluated_data["true_labels"].cpu().detach().numpy()
    test_score = aggregated_evaluated_data["test_score"].cpu().detach().numpy()
    test_pred = aggregated_evaluated_data["test_pred"].cpu().detach().numpy()
    order = np.argsort(-test_score)

    top_2000 = test_score[order[1:200]]
    top_2000_labels = true_labels[order[1:200]]

    act = top_2000[top_2000_labels == 0]
    inact = top_2000[top_2000_labels == 1]

    evaluated_df = pd.DataFrame({
        "Score": test_score,
        "GT": true_labels,
        "Pred": test_pred,
        "Indices": train_test_dataset.test_dataset.indices
    })

    evaluated_df.to_csv(save_path + "/eval.csv", index=False)
    print(f"Model saved to {save_path}")

    return {
        "evaluated_df": evaluated_df,
        "test_score": test_score,
        "test_pred": test_pred,
        "true_labels": true_labels,
        "act": act,
        "inact": inact
    }

@torch.no_grad
def embeddings_evaluate_function(model, data, agg_dict, device=torch.device("cpu")):
    margin = 1
    pos_to_use = [42, 44, 61, 65, 68, 69, 72, 94, 108, 112, 121, 145, 148, 150, 167, 181, 185, 203, 205, 220, 222, 224]
    x = data[0].to(device)
    y = data[1].to(device)

    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)

    hh = model(x)

    emb = torch.nn.functional.normalize(hh[:, torch.tensor(pos_to_use), :], dim=1).mean(dim=1)
    emb = torch.nn.functional.normalize(emb, dim=1)

    if "trip_loss" not in agg_dict.keys():
        agg_dict['trip_loss'] = torch.tensor([], dtype=torch.float, device=device)
    
    if "embeddings" not in agg_dict.keys():
        agg_dict['embeddings'] = torch.tensor([], dtype=torch.float, device=device)

    if "ground_truth" not in agg_dict.keys():
        agg_dict['ground_truth'] = torch.tensor([], dtype=torch.float, device=device)

    trips = torch.tensor(online_mine_triplets(y))

    if len(trips) > 0:
        emb_trip = emb[trips]
        trip_loss = triplet_loss(emb_trip[:, 0, :], emb_trip[:, 1, :], emb_trip[:, 2, :])
        agg_dict["trip_loss"] = torch.cat([agg_dict['trip_loss'], trip_loss.detach().reshape(-1)], dim=0)

    agg_dict["embeddings"] = torch.cat([agg_dict['embeddings'], emb.detach()], dim=0)
    agg_dict["ground_truth"] = torch.cat([agg_dict["ground_truth"], y.detach().reshape(-1)], dim=0)

    return agg_dict

@torch.no_grad()
def embeddings_finalize_function(agg_dict, dataset):
    embeddings = agg_dict["embeddings"].cpu().numpy()
    ground_truth = agg_dict["ground_truth"].cpu().numpy()

    # If embeddings is bigger than 15K, randomly sample 15K data points (by value, not by reference)
    max_points = 15000
    n_points = embeddings.shape[0]
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        embeddings = embeddings[idx].copy()
        ground_truth = ground_truth[idx].copy()

    tsne = TSNE(n_components=3, random_state=42)
    emb_3d = tsne.fit_transform(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=ground_truth, cmap='viridis', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    ax.set_title("t-SNE of Embeddings (3D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    plt.show()

    # Instead of showing the plot, return the figure object so it can be saved elsewhere
    plt.close(fig)  # Close the figure to prevent it from displaying in some environments
    agg_dict["tsne_visualization"] = fig

    # Compute pairwise cosine distance for all embeddings (no subsampling)
    all_embeddings = agg_dict["embeddings"].cpu()
    all_labels = agg_dict["ground_truth"].cpu().numpy()
    # Use the pairwise_cosine function defined at the top of the file
    distance_matrix = pairwise_cosine(all_embeddings).cpu().numpy()

    avg_dist_to_1 = []
    avg_dist_to_0 = []
    actual_label = []

    for i in range(distance_matrix.shape[0]):
        label_i = all_labels[i]
        # Indices for label 1 and label 0 (excluding self)
        idx_1 = np.where((all_labels == 1) & (np.arange(len(all_labels)) != i))[0]
        idx_0 = np.where((all_labels == 0) & (np.arange(len(all_labels)) != i))[0]
        # Average distances
        avg1 = np.mean(distance_matrix[i, idx_1]) if len(idx_1) > 0 else np.nan
        avg0 = np.mean(distance_matrix[i, idx_0]) if len(idx_0) > 0 else np.nan
        avg_dist_to_1.append(avg1)
        avg_dist_to_0.append(avg0)
        actual_label.append(label_i)

    df = pd.DataFrame({
        "avg_dist_to_1": avg_dist_to_1,
        "avg_dist_to_0": avg_dist_to_0,
        "actual_label": actual_label,
        "indices": np.array(dataset.indices)
    })
    
    # Overlay histograms of avg_dist_to_1 and avg_dist_to_0 for label 0 and label 1 separately

    # Subset for label 0
    df_0 = df[df["actual_label"] == 0]
    # Subset for label 1
    df_1 = df[df["actual_label"] == 1]

    # Plot for label 0: overlay avg_dist_to_1 and avg_dist_to_0
    fig0, ax0 = plt.subplots(figsize=(7, 5))
    ax0.hist(df_0["avg_dist_to_1"].dropna(), bins=30, alpha=0.6, color='gray', label='Avg Dist to negatives')
    ax0.hist(df_0["avg_dist_to_0"].dropna(), bins=30, alpha=0.6, color='green', label='Avg Dist to positives')
    ax0.set_title("Positives: Avg Dist to negative and positives (Overlayed)")
    ax0.set_xlabel("Average Distance")
    ax0.set_ylabel("Frequency")
    ax0.legend()
    plt.tight_layout()

    agg_dict["hist_overlay_avg_dist_label0"] = fig0
    plt.close(fig0)

    # Plot for label 1: overlay avg_dist_to_1 and avg_dist_to_0
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.hist(df_1["avg_dist_to_1"].dropna(), bins=30, alpha=0.6, color='gray', label='Avg Dist to negatives')
    ax1.hist(df_1["avg_dist_to_0"].dropna(), bins=30, alpha=0.6, color='green', label='Avg Dist to positives')
    ax1.set_title("Negatives: Avg Dist to negatives and positives (Overlayed)")
    ax1.set_xlabel("Average Distance")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    plt.tight_layout()
    agg_dict["hist_overlay_avg_dist_label1"] = fig1
    plt.close(fig1)
    agg_dict["pairwise_distance_df"] = df

    return agg_dict


# Old code (now replaced by YAML config):
# root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
# dataset_path = f"{root_path}/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv"
# save_path = f"{root_path}/pretraining/triplet_loss_backbones/one_shot/" 
# weights_path = f"{root_path}/pretraining/triplet_loss_backbones/final_model.pt"

ref_seq = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"

plm_name = "esm2_t12_35M_UR50D"
train_indices_func = lambda sdf: get_indices(sdf, [1, 2, 3], nmuts_column="num_muts")
test_indices_func = lambda sdf: get_indices(sdf, [1, 2, 3], nmuts_column="num_muts", rev=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = plmTrunkModel(
    plm_name=plm_name,
    opmode="mean",
    specific_pos=None,
    hidden_layers=[4096,4096],
    activation="sigmoid",
    layer_norm=False,
    activation_on_last_layer=False,
    device=device
).to(device)

# Load weights for plm backbone only
backbone_weights = torch.load(weights_path, map_location=device)
backbone_weights = {k.replace("plm.", "", 1): v for k, v in backbone_weights.items() if "plm." in k}
model.plm.load_state_dict(backbone_weights, strict=True)

train_test_dataset = EpiNNetActivityTrainTest(
    train_project_name="triplet_training",
    evaluation_path=save_path,
    dataset_path=dataset_path,
    train_indices=train_indices_func,
    test_indices=test_indices_func,
    encoding_function=model.encode,
    encoding_identifier=plm_name,
    cache=True,
    lazy_load=True,
    sequence_column_name='full_seq',
    activity_column_name='inactive',
    ref_seq=ref_seq,
    labels_dtype=torch.float32,
    device=device
)

# #Train
# model = \
#     train_plm_trunk(
#         plm_name=plm_name,
#         save_path=save_path,
#         train_test_dataset=train_test_dataset,
#         device=device,
#         model=model, 
#         lr=1e-6,
#     )

train_test_dataset.evaluate(model,
                            embeddings_evaluate_function, 
                            embeddings_finalize_function,
                            eval_train=False)


# EPINNET DATASET:

    # print("Preparing dataset...")
    # train_test_dataset = EpiNNetActivityTrainTest(train_project_name="triplet_training",
    #                                               evaluation_path="",
    #                                               dataset_path=dataset_path,
    #                                               train_indices=train_indices_func,
    #                                               test_indices=test_indices_func,
    #                                               encoding_function=None,
    #                                               encoding_identifier=None,
    #                                               external_encoding=encodings,
    #                                               cache=True,
    #                                               lazy_load=True,
    #                                               sequence_column_name='full_seq',
    #                                               activity_column_name='inactive',
    #                                               ref_seq=ref_seq,
    #                                               labels_dtype=torch.float32,
    #                                               device=device)



    