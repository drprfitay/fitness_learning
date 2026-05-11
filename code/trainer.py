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
from torch.utils.data import Subset
import loralib as lora
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import einops


import time


from utils import *
from dataset import *
from embedder import *
from plm_base import *




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
    

def select_design_pos(seq):
    return([seq[21], seq[23], seq[24]])

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
    opmode="pos",
    hidden_layers=[1024],
    activation="sigmoid",
    train_type="triplet",
    layer_norm=False,
    activation_on_last_layer=False,
    device=torch.device("cpu"),
    model=None  
):

    print("\n[DEBUG] Training parameters:")
    print(f"\tplm_name: {plm_name}")
    print(f"\tsave_path: {save_path}")
    print("\ttrain_dataset size: %d" % len(train_test_dataset.train_dataset))
    print(f"\tpos_to_use: {pos_to_use}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\titerations: {iterations}")
    print(f"\tmargin: {margin}")
    print(f"\tlr: {lr}")
    print(f"\tweight_decay: {weight_decay}")
    print(f"\tencoding_identifier: {encoding_identifier}")
    print(f"\topmode: {opmode}")
    print(f"\thidden_layers: {hidden_layers}")
    print(f"\tactivation: {activation}")
    
    print(f"\tlayer_norm: {layer_norm}")
    print(f"\tactivation_on_last_layer: {activation_on_last_layer}")
    print(f"\tdevice: {device}")
    print(f"\tmodel: {type(model)}\n")
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoints_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # In the future we might have other use cases,
    emb_only = train_type  == "triplet"

    if model is None:
        model = plmEmbeddingModel(
            plm_name=plm_name,
            opmode=opmode,
            emb_only=emb_only,
            specific_pos=pos_to_use,
            hidden_layers=hidden_layers,
            activation=activation,
            layer_norm=layer_norm,
            activation_on_last_layer=activation_on_last_layer,
            device=device
        ).to(device)
    else:
        model = model.to(device)

    if train_type == "direct_mlp":
        for layer in model.epinnet_trunk.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    if train_test_dataset is None:
        raise ValueError("train_test_dataset must be provided as an argument.")

    if pos_to_use is None:
        pos_to_use = [int(x[1:]) for x in train_test_dataset.train_dataset.sequence_dataframe.columns[3:25].tolist()]

    print(f"Using positions: {pos_to_use}")
    train_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    avg_loss = torch.tensor([]).to(device)
    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    train_test_dataset.train_dataset.labels = torch.nn.functional.one_hot(
        train_test_dataset.train_dataset.labels.to(torch.long), 2
    ).to(torch.float)

    # Freeze all layers for the plm but the last one
    if train_type == "direct_mlp":
        if hasattr(model, "plm") and hasattr(model.plm, "layers"):
            for i, layer in enumerate(model.plm.layers):
                for param in layer.parameters():
                    param.requires_grad = (i == len(model.plm.layers) - 1)


    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)

            if train_type == "direct_mlp":
                optimizer.zero_grad()
                a = model(x)
                total_loss = ce_loss_fn(a[2], y)
            elif train_type == "triplet":
                trips = torch.tensor(online_mine_triplets(y))

                if len(trips) <= 0:
                    continue                     

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
            #print("[E%d I%d] %.3f { Triplet :%.3f}" % (epoch, step, total_loss, trip_loss))
            # print(torch.unique(a[2].softmax(dim=1).argmax(dim=1), return_counts=True))
            print("[E%d I%d] %.3f " % (epoch, step, total_loss))
            if total_steps % 1000 == 0:
                print("\t\tCheckpoint [%d]" % total_steps)
                torch.save(model.state_dict(), checkpoints_dir + "/checkpoint_model_%d.pt" % total_steps)
                torch.save(running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt")
                torch.save(running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt")
                torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")
        running_epoch_loss = torch.cat([running_epoch_loss, epoch_loss.detach().reshape(-1)])
    torch.save(model.state_dict(), save_path + "/final_model.pt")
    torch.save(running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model

def train_epinnet(
    train_test_dataset,
    save_path: str,
    encodings=None,
    model=None,
    batch_size=32,
    iterations=20000,                  
    lr=1e-4,
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
                plt.plot(range(1, running_20b_loss.shape[0] + 1), running_20b_loss.cpu().detach().numpy())
                plt.draw()
                plt.pause(0.001)
                plt.close()
                
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
    
    # REVERT BACK LABELS
    train_test_dataset.train_dataset.labels = train_test_dataset.train_dataset.labels.argmax(dim=1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path + "/final_model.pt")
    torch.save(running_batch_loss.cpu().detach(), save_path + "batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), save_path + "epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), save_path + "20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model

def train_msa_backbone(
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
    opmode="pos",
    hidden_layers=[1024],
    activation="sigmoid",
    layer_norm=False,
    activation_on_last_layer=False,
    device=torch.device("cpu"),
    model=None  
):

    print("\n[DEBUG] Training parameters:")
    print(f"\tplm_name: {plm_name}")
    print(f"\tsave_path: {save_path}")
    print("\ttrain_dataset size: %d" % len(train_test_dataset.train_dataset))
    print(f"\tpos_to_use: {pos_to_use}")
    print(f"\tbatch_size: {batch_size}")
    print(f"\titerations: {iterations}")
    print(f"\tmargin: {margin}")
    print(f"\tlr: {lr}")
    print(f"\tweight_decay: {weight_decay}")
    print(f"\tencoding_identifier: {encoding_identifier}")
    print(f"\topmode: {opmode}")
    print(f"\thidden_layers: {hidden_layers}")
    print(f"\tactivation: {activation}")
    
    print(f"\tlayer_norm: {layer_norm}")
    print(f"\tactivation_on_last_layer: {activation_on_last_layer}")
    print(f"\tdevice: {device}")
    print(f"\tmodel: {type(model)}\n")
    torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoints_dir = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # In the future we might have other use cases,
    logits_only = True

    if model is None:
        model = plmEmbeddingModel(
            plm_name=plm_name,
            emb_only=False
            # opmode=opmode,
            # logits_only=logits_only,
            # hidden_layers=hidden_layers,
            # activation=activation,
            # emb_only=False,
            # layer_norm=layer_norm,
            # activation_on_last_layer=activation_on_last_layer,
            # device=device
        ).to(device)
    else:
        model = model.to(device)

    if train_test_dataset is None:
        raise ValueError("train_test_dataset must be provided as an argument.")

    train_loader = torch.utils.data.DataLoader(train_test_dataset, batch_size=batch_size, shuffle=True)
    n_epochs = ceil(iterations / len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-66)

    model.train()
    avg_loss = torch.tensor([]).to(device)
    total_steps = 0
    running_batch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_epoch_loss = torch.tensor([], dtype=torch.float).to(device)
    running_20b_loss = torch.tensor([], dtype=torch.float).to(device)

    def noise_schedule(ranges, p=0.1):
        head_positions = []
        for r in ranges:
            r_list = list(r)
            n = len(r_list)
            if n == 0:
                continue
            probs = torch.tensor([1 - p, p])
            tosses = torch.multinomial(probs.repeat(n, 1), 1).squeeze(1)
            # Get positions where toss == 1 (heads)
            heads = [r_list[i] for i in range(n) if tosses[i].item() == 1]
            head_positions.extend(heads)
        return torch.tensor(head_positions, dtype=torch.long)

    ongoing_range_dict = {}

    def mine_ranges(ranges, indices):

        return_range_list = []
        for idx, global_idx in enumerate(indices):
            if global_idx in ongoing_range_dict:
                return_range_list.append(ongoing_range_dict[global_idx])
            else:
                rng = ranges[idx]
                rng = [x.split("_") for x in rng.split("x")]
                rng = [range(int(x[0]), int(x[1])) if len(x) == 2 else range(int(x[0]), int(x[0]) + 1) for x in rng]
                return_range_list.append(rng)
                ongoing_range_dict[global_idx] = rng

        return return_range_list


    mask_token = torch.tensor(model.tokenizer.encode("<mask>")).to(device)
        
    for epoch in range(n_epochs):
        epoch_loss = torch.tensor(0.0).to(device)
        iter_20b_loss = torch.tensor(0.0).to(device)
        for step, batch in enumerate(train_loader):
            x = batch[0].to(device)
            y = batch[1].to(device)                  
            # Ensure y is on CPU before converting to numpy, for CUDA compatibility
            y_cpu = y.detach().cpu().numpy()

            pad_regions = train_test_dataset.train_dataset.get_pad_regions()[y_cpu].tolist()

            ranges = mine_ranges(
                #train_test_dataset.train_dataset.sequence_dataframe["pad_regions"].iloc[y_cpu].to_list(),
                pad_regions,
                y_cpu
            )
            # Apply random noise to the sequence
            mask_pos_matrix = torch.stack([torch.nn.functional.one_hot(noise_schedule(rng), x.shape[1]).sum(dim=0) for rng in ranges], dim=0)
            mask_pos_matrix = mask_pos_matrix.to(device)

            # Place masks instead of noise
            masked_sequence = (x - x * mask_pos_matrix) + mask_pos_matrix * mask_token

            logits = model(masked_sequence)

            # Get the ground truth labels on masked positiosn
            gt_labels = (((torch.ones(x.shape, device=device) - mask_pos_matrix) * -66) + (x * mask_pos_matrix)).view(-1)
            total_loss = ce_loss_fn(logits.view(-1, len(model.vocab)), gt_labels.to(torch.long))

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
            #print("[E%d I%d] %.3f { Triplet :%.3f}" % (epoch, step, total_loss, trip_loss))
            # print(torch.unique(a[2].softmax(dim=1).argmax(dim=1), return_counts=True))
            print("[E%d I%d] %.3f " % (epoch, step, total_loss))
            if total_steps % 1000 == 0:
                print("\t\tCheckpoint [%d]" % total_steps)
                torch.save(model.state_dict(), checkpoints_dir + "/checkpoint_model_%d.pt" % total_steps)
                torch.save(running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt")
                torch.save(running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt")
                torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")

        running_epoch_loss = torch.cat([running_epoch_loss, epoch_loss.detach().reshape(-1)])

    torch.save(model.state_dict(), save_path + "/final_model.pt")
    torch.save(running_batch_loss.cpu().detach(), save_path + "/batch_loss.pt")
    torch.save(running_epoch_loss.cpu().detach(), save_path + "/epoch_loss.pt")
    torch.save(running_20b_loss.cpu().detach(), save_path + "/20b_loss.pt")
    print(f"Model saved to {save_path}")
    return model

def train_evaluate_plms(config):
    ref_seq = config["ref_seq"]

    # Check if "plm_name" exists in config, else use default "esm2_t12_35M_UR50D"
    if "plm_name" in config and config["plm_name"]:
        plm_name = config["plm_name"]
    else:
        print("Warning: 'plm_name' not found in config. Using default 'esm2_t12_35M_UR50D'.")
        plm_name = "esm2_t12_35M_UR50D"


    train_indices_func = lambda sdf: get_indices(
        sdf, 
        config["train_indices"], 
        nmuts_column=config["nmuts_column"],
        rev=config["train_indices_rev"]
    )

    test_indices_func = lambda sdf: get_indices(
        sdf, 
        config["test_indices"], 
        nmuts_column=config["nmuts_column"],
        rev=config["test_indices_rev"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"ֿ\t\t[INFO] Using device: {device}")

    emb_only = config["train_type"] == "triplet"
    
    model = plmEmbeddingModel(
        plm_name=plm_name,
        emb_only = emb_only
        # opmode="pos",
        # hidden_layers=[516,256],
        # activation="relu",
        # layer_norm=False,
        # activation_on_last_layer=False,
        # specific_pos=config["pos_to_use"],
        # device=device
    ).to(device)

    # Load weights for plm backbone only if config has load_weights set to True
    if "load_weights" in config and config["load_weights"]:
        backbone_weights = torch.load(config["weights_path"], map_location=device)
        backbone_weights = {k.replace("plm.", "", 1): v for k, v in backbone_weights.items() if "plm." in k}
        model.plm.load_state_dict(backbone_weights, strict=True)

    train_test_dataset = PREActivityDataset(
        train_project_name="triplet_training",
        evaluation_path=config["save_path"],
        dataset_path=config["dataset_path"],
        train_indices=train_indices_func,
        test_indices=test_indices_func,
        encoding_function=model.encode,
        encoding_identifier=plm_name,
        cache=True,
        lazy_load=True,
        sequence_column_name=config["sequence_column_name"],
        label_column_name=config["label_column_name"],
        ref_seq=ref_seq,
        labels_dtype=torch.float32,
        device=device
    )

    if config["train"]:

        model.plm.token_dropout = config["train_drop_tokens"]
        
        model = \
            train_plm_triplet_model(
                plm_name=plm_name,
                train_type=config["train_type"],
                save_path=config["save_path"],
                train_test_dataset=train_test_dataset,
                pos_to_use=config["pos_to_use"],
                batch_size=config["batch_size"],
                iterations=config["iterations"],
                lr=float(config["lr"]),
                device=device,
                model=model, 
            )

    model.plm.token_dropout = config["inference_drop_tokens"]

    if config["evaluate_train"] or config["evaluate_test"]:
        train_test_dataset.evaluate(model,
                                    embeddings_evaluate_function, 
                                    embeddings_finalize_function,
                                    eval_train=config["evaluate_train"],
                                    eval_test=config["evaluate_test"],
                                    internal_batch_size=config["batch_size"])

# EPINNET DATASET:
def train_evaluate_epinnet(config):

    # Define train/test split functions
    train_indices_func = lambda sdf: get_indices(
        sdf, 
        config["train_indices"], 
        nmuts_column=config["nmuts_column"],
        rev=config["train_indices_rev"]
    )
    
    test_indices_func = lambda sdf: get_indices(
        sdf, 
        config["test_indices"], 
        nmuts_column=config["nmuts_column"],
        rev=config["test_indices_rev"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the sequence dataframe and create encodings
    sequence_df = pd.read_csv(config["dataset_path"])
    encodings = torch.tensor(get_one_hot_encoding(sequence_df, config["first_column_name"], config["last_column_name"]), dtype=torch.float)

    # Initialize EpiNNet model
    epinnet_model = EpiNNet(
        d_in=encodings.shape[1],  # Example input dimension, adjust as needed
        hidden_layers=[512, 256],  # Example hidden layers, adjust as needed
        d_out=2,  # Example output dimension for binary classification
        activation="relu",
        device=device
    ).to(device)
    

    if "load_weights" in config and config["load_weights"]:
        backbone_weights = torch.load(config["weights_path"], map_location=device)
        epinnet_model.load_state_dict(backbone_weights)

    train_test_dataset = PREActivityDataset(
        train_project_name="triplet_training",
        evaluation_path=config["save_path"],
        dataset_path=config["dataset_path"],
        train_indices=train_indices_func,
        test_indices=test_indices_func,
        encoding_function=None,
        encoding_identifier=None,
        external_encoding=encodings,
        cache=True,
        lazy_load=True,
        sequence_column_name='full_seq',
        label_column_name='inactive',
        ref_seq=config["ref_seq"],
        labels_dtype=torch.float32,
        device=device
    )

    if config["train"]:

        epinnet_model =\
            train_epinnet(
                train_test_dataset=train_test_dataset,
                save_path=config["save_path"],
                encodings=encodings,
                model=epinnet_model,
                device=device,
                lr=float(config["lr"]),
                iterations=int(config["iterations"]),
                batch_size=int(config["batch_size"]),
            )

        
    if config["evaluate_train"] or config["evaluate_test"]:
        train_test_dataset.evaluate(epinnet_model,   
                                    epinnet_evaluate_function,
                                    epinnet_finalize_function,
                                    eval_train=config["evaluate_train"],
                                    eval_test=config["evaluate_test"])

def train_evaluate_msa_backbone(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if "plm_name" exists in config, else use default "esm2_t12_35M_UR50D"
    if "plm_name" in config and config["plm_name"]:
        plm_name = config["plm_name"]
    else:
        print("Warning: 'plm_name' not found in config. Using default 'esm2_t12_35M_UR50D'.")
        plm_name = "esm2_t12_35M_UR50D"

    msa_backbone_model =\
     plmEmbeddingModel(
        plm_name=plm_name,
        emb_only = False
        # opmode="pos",
        # logits_only = True,
        # hidden_layers=[516,256],
        # activation="relu",
        # layer_norm=False,
        # activation_on_last_layer=False,
        # device=device
    ).to(device)


    def encode_aligned_seq(seq):
        seq = seq.replace("-", "<pad>")
        return msa_backbone_model.tokenizer.encode(seq)

    # Prepare dataset (assuming similar interface as PREActivityDataset)
    train_test_dataset = PREActivityDataset(
        train_project_name="triplet_training",
        evaluation_path=config["save_path"],
        dataset_path=config["dataset_path"],
        train_indices=None,
        test_indices=None,
        encoding_function=encode_aligned_seq,
        encoding_identifier=plm_name,
        cache=True,
        lazy_load=True,
        sequence_column_name=config["sequence_column_name"],
        pad_region_label=config["pad_region_label"],
        label_column_name=None, # This should be none so the labels would be just the indices
        ref_seq=config["ref_seq"],
        labels_dtype=torch.float32,
        device=device
    )

    if config["train"]:
        msa_backbone_model.train()
        msa_backbone_model = train_msa_backbone(
                plm_name=plm_name,
                train_test_dataset=train_test_dataset,
                save_path=config["save_path"],                
                batch_size=config["batch_size"],
                iterations=config["iterations"],
                lr=float(config["lr"]),
                device=device,
                model=msa_backbone_model, 
            )


    # TODO: Add evaluation function   
    # if config["evaluate_train"] or config["evaluate_test"]:
    #     train_test_dataset.evaluate(
    #         msa_backbone_model,
    #         #msa_backbone_evaluate_function,
    #         #msa_backbone_finalize_function,
    #         eval_train=config.get("evaluate_train", False),
    #         eval_test=config.get("evaluate_test", False)
    #     )





