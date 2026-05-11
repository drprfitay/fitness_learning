
import sys
import torch
import os
import shutil
import subprocess
import glob
import gzip
import pandas as pd 
import numpy as np
import fcntl
import time
import random

from utils import *

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


@torch.no_grad()
def ab_embedding_evaluate_heavy_or_light_chain(model, data, agg_dict, device=torch.device("cpu"), **kwargs):
    batch = data[0]
    model = model.to(device)
    batch = batch.to(device)



    heavy_or_light_only = kwargs["heavy_or_light_only"] if "heavy_or_light_only" in kwargs else "heavy_only"
    separator_location = batch == 3
    separator_location_idx = torch.where(separator_location[0])[0][0]

    if heavy_or_light_only == "heavy_only":
        batch = batch[:,0:(separator_location_idx + 1)] # Everything before the separator + separator token (which is same as EOS)
    elif heavy_or_light_only == "light_only":
        bos_token = batch[0,0]
        batch = batch[:,separator_location_idx:]
        batch[:,0] = bos_token # update separator token to be the bos token
    else:
        raise ValueError(f"Invalid HEAVY_OR_LIGHT_ONLY value: {heavy_or_light_only}")

    # We want the pads to be removed from the attention mask
    pads = batch == 0
    attention_mask = torch.ones(batch.shape, dtype=torch.int, device=device) - pads.to(torch.int)

    embeddings = model(batch, attention_mask=attention_mask)

    B, S, D = embeddings.shape

    # Remove 1st / last token
    batch = batch[:, 1:-1]
    embeddings = embeddings[:, 1:-1, :]
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    tokens = batch
    tokens_to_include = (tokens != 0).to(torch.int32)

    N_used_tokens = tokens_to_include.sum(dim=1).unsqueeze(1)
    embeddings_without_pad = (embeddings * tokens_to_include.unsqueeze(dim=2)).sum(dim=1)
    embeddings_final = embeddings_without_pad / N_used_tokens
    embeddings_final = torch.nn.functional.normalize(embeddings_final, dim=1)

    if heavy_or_light_only not in agg_dict.keys():
        agg_dict[heavy_or_light_only] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))

    agg_dict[heavy_or_light_only] = torch.cat([agg_dict[heavy_or_light_only], embeddings_final.detach().cpu()], dim=0)

    return agg_dict

@torch.no_grad()
def ab_embeddings_evaluate_function(model, data, agg_dict, device=torch.device("cpu"), **kwargs):
    batch = data[0]
    model = model.to(device)
    batch = batch.to(device)
    
    # We want the pads to be removed from the attention mask
    pads = batch == 0
    attention_mask = torch.ones(batch.shape, dtype=torch.int, device=device) - pads.to(torch.int)

    embeddings = model(batch, attention_mask=attention_mask)

    B, S, D = embeddings.shape

    # Remove 1st / last token
    batch = batch[:, 1:-1]
    embeddings = embeddings[:, 1:-1, :]

    separator_location = batch == 3
    separator_location_idx = torch.where(separator_location[0])[0]

    h_chain_tokens = batch[:,0:separator_location_idx]
    l_chain_tokens = batch[:,separator_location_idx+1:]

    h_chain_embeddings = torch.nn.functional.normalize(embeddings[:,0:separator_location_idx,:], dim=1)
    l_chain_embeddings = torch.nn.functional.normalize(embeddings[:,separator_location_idx+1:,:], dim=1)
    
    h_chain_tokens_to_include = (h_chain_tokens != 0).to(torch.int32)
    l_chain_tokens_to_include = (l_chain_tokens != 0).to(torch.int32)

    N_used_tokens_h = h_chain_tokens_to_include.sum(dim=1).unsqueeze(1)
    N_used_tokens_l = l_chain_tokens_to_include.sum(dim=1).unsqueeze(1)

    h_chain_embeddings_without_pad = (h_chain_embeddings * h_chain_tokens_to_include.unsqueeze(dim=2)).sum(dim=1)
    l_chain_embeddings_without_pad = (l_chain_embeddings * l_chain_tokens_to_include.unsqueeze(dim=2)).sum(dim=1)

    h_chain_embeding_final = h_chain_embeddings_without_pad / N_used_tokens_h
    l_chain_embeddings_final = l_chain_embeddings_without_pad / N_used_tokens_l
    joint_final_embeddings = (h_chain_embeddings_without_pad + l_chain_embeddings_without_pad) / (N_used_tokens_h + N_used_tokens_l)

    h_chain_embeddings_final = torch.nn.functional.normalize(h_chain_embeding_final, dim=1)
    l_chain_embeddings_final = torch.nn.functional.normalize(l_chain_embeddings_final, dim=1)
    joint_embeddings_final = torch.nn.functional.normalize(joint_final_embeddings, dim=1)

    if "h_chain" not in agg_dict.keys():
        agg_dict['h_chain'] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
    
    if "l_chain" not in agg_dict.keys():
        agg_dict['l_chain'] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))

    if "joint" not in agg_dict.keys():
        agg_dict['joint'] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))

    agg_dict["h_chain"] = torch.cat([agg_dict['h_chain'], h_chain_embeddings_final.detach().cpu()], dim=0)
    agg_dict["l_chain"] = torch.cat([agg_dict['l_chain'], l_chain_embeddings_final.detach().cpu()], dim=0)
    agg_dict["joint"] = torch.cat([agg_dict['joint'], joint_embeddings_final.detach().cpu()], dim=0)

    return agg_dict

@torch.no_grad()
def ab_embeddings_finalize_function(agg_dict, dataset, **kwargs):
    return agg_dict

@torch.no_grad()
def epinnet_evaluate_function(model, data, aggregated_evaluated_data, device=torch.device("cpu"), **kwargs):
    x = data[0].to(device)
    y = data[1].to(device)
    y = torch.nn.functional.one_hot(
       y.to(torch.long), 2
    ).to(torch.float)
    y_pred = model(x)
    ce_loss_fn = torch.nn.CrossEntropyLoss()
    batch_loss = ce_loss_fn(y_pred, y)

        # Accumulate predictions, scores, true labels, and loss
    if "pred_label" not in aggregated_evaluated_data:
        aggregated_evaluated_data["pred_label"] = torch.tensor([], dtype=torch.long)
    if "active_prob" not in aggregated_evaluated_data:
        aggregated_evaluated_data["active_prob"] = torch.tensor([], dtype=torch.float)
    if "true_label" not in aggregated_evaluated_data:
        aggregated_evaluated_data["true_label"] = torch.tensor([], dtype=torch.long)
    if "loss" not in aggregated_evaluated_data:
        aggregated_evaluated_data["loss"] = torch.tensor([], dtype=torch.float)

    aggregated_evaluated_data["pred_label"] = torch.cat(
        [aggregated_evaluated_data["pred_label"], y_pred.argmax(dim=1).cpu().detach()], dim=0
    )
    aggregated_evaluated_data["active_prob"] = torch.cat(
        [aggregated_evaluated_data["active_prob"], y_pred.softmax(dim=1)[:, 0].cpu().detach()]
    )
    aggregated_evaluated_data["true_label"] = torch.cat(
        [aggregated_evaluated_data["true_label"], y.argmax(dim=1).cpu().detach()]
    )
    aggregated_evaluated_data["loss"] = torch.cat(
        [aggregated_evaluated_data["loss"], batch_loss.detach().reshape(-1).cpu()]
    )

    # Flush the CUDA cache to free up unused memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return aggregated_evaluated_data

@torch.no_grad()
def epinnet_finalize_function(aggregated_evaluated_data, dataset):
    # Restore original finalize logic
    true_label = aggregated_evaluated_data["true_label"].cpu().detach()
    active_prob = aggregated_evaluated_data["active_prob"].cpu().detach()
    pred_label = aggregated_evaluated_data["pred_label"].cpu().detach()

    evaluated_df = pd.DataFrame({
        "Score": active_prob,
        "GT": true_label,
        "Pred": pred_label,
        "Indices": dataset.indices
    })

    #top_K_df = pd.DataFrame(dict([("%d" % K, np.unique(true_label[np.argsort(-active_prob)[0:K]], return_counts=True)[1]) for K in [5,10,50,100,500,1000,5000]]))

    return {
        "evaluated_df": evaluated_df,
        "predicted_score": active_prob,
        "predicted_label": pred_label,
        "true_label": true_label,
        #"top_K_df": top_K_df,
    }

@torch.no_grad()
def embeddings_evaluate_function(model, data, agg_dict, device=torch.device("cpu"), **kwargs):
    margin = 1
    
    x = data[0].to(device)

    print("[DBG]########################")
    print("x.shape: ", x.shape)
    print(x.sum(dim=1))
    print("[DBG]########################")
    y = data[1].to(device)

    calc_triplets = "triplet_loss" in kwargs

    if calc_triplets:
        triplet_loss = torch.nn.TripletMarginLoss(margin=margin, eps=1e-7)

    hh = model(x)

    
    if "positions_to_embed" in kwargs and kwargs["positions_to_embed"] is not None:
        pos_to_use = kwargs["positions_to_embed"]
    else:
        pos_to_use = list(range(1, hh.shape[1] - 1)) # ignore first and last token [bos/eos]

    if "normalize_embeddings" in kwargs:
        normalize_embeddings = kwargs["normalize_embeddings"]
    else:
        normalize_embeddings = True

    if normalize_embeddings:
        print("[INFO]: Normalizing embeddings")
        emb = torch.nn.functional.normalize(hh[:, torch.tensor(pos_to_use), :], dim=1)

    if "average_embeddings" in kwargs and kwargs["average_embeddings"]:
        print("[INFO]: Averaging embeddings")
        emb.mean(dim=1)

        if normalize_embeddings:
            print("[INFO]: Normalizing averaged embeddings")
            emb = torch.nn.functional.normalize(emb, dim=1)

    if calc_triplets and "trip_loss" not in agg_dict.keys():
        agg_dict['trip_loss'] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))
    
    if "embeddings" not in agg_dict.keys():
        agg_dict['embeddings'] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))

    if "y_value" not in agg_dict.keys():
        agg_dict['y_value'] = torch.tensor([], dtype=torch.float, device=torch.device("cpu"))


    if calc_triplets:
        trips = torch.tensor(online_mine_triplets(y))
        if len(trips) > 0:
            emb_trip = emb[trips]
            trip_loss = triplet_loss(emb_trip[:, 0, :], emb_trip[:, 1, :], emb_trip[:, 2, :])
            agg_dict["trip_loss"] = torch.cat([agg_dict['trip_loss'], trip_loss.detach().cpu().reshape(-1)], dim=0)

    agg_dict["embeddings"] = torch.cat([agg_dict['embeddings'], emb.detach().cpu()], dim=0)
    agg_dict["y_value"] = torch.cat([agg_dict["y_value"], y.detach().cpu().reshape(-1)], dim=0)

    torch.cuda.empty_cache() 
    
    return agg_dict

@torch.no_grad()
def embeddings_finalize_function(agg_dict, dataset, **kwargs):
    return agg_dict

@torch.no_grad()
def tsne_embeddings_finalize_function(agg_dict, dataset):
    embeddings = agg_dict["embeddings"].cpu().numpy()
    ground_truth = agg_dict["ground_truth"].cpu().numpy()

    # If embeddings is bigger than 15K, randomly sample 15K data points (by value, not by reference)
    max_points = 15000
    n_points = embeddings.shape[0]
    if n_points > max_points:
        idx = np.random.choice(n_points, max_points, replace=False)
        embeddings = embeddings[idx].copy()
        ground_truth = ground_truth[idx].copy()

    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=ground_truth, cmap='viridis', alpha=0.7)
    legend1 = ax.legend(*scatter.legend_elements(), title="Labels")
    ax.add_artist(legend1)
    ax.set_title("t-SNE of Embeddings (2D)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
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

