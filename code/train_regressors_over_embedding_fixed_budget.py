import argparse
import pandas as pd
import numpy as np
import os
import torch
import pickle

from sklearn.neural_network import MLPRegressor
from scipy.stats import spearmanr

base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)
    return one_hot_encoding

def load_nmt():
    df = pd.read_csv("%s/data/nmt/nmt_full_seq.csv" % base_path)
    embedding_paths = {
        "progen2-small": "%s/data/nmt/embeddings/progen2-small" % base_path,
        "progen2-medium": "%s/data/nmt/embeddings/progen2-medium" % base_path,
        "esm_35m": "%s/data/nmt/embeddings/esm_35m" % base_path,
        "esm_8m": "%s/data/nmt/embeddings/esm_8m" % base_path,
        "esm_3b": "%s/data/nmt/embeddings/esm_3b" % base_path,
        "esm_650m": "%s/data/nmt/embeddings/esm_650m" % base_path,
        "prot_bert": "%s/data/nmt/embeddings/prot_bert" % base_path,
        "esm_150m": "%s/data/nmt/embeddings/esm_150m" % base_path,
    }
    si = np.where(df.columns == "1")[0][0]
    ei = np.where(df.columns == "272")[0][0]+1
    positions_with_mutations =  np.array([len(pd.unique(df.iloc[:,i])) > 1 for i in range(si,ei)])
    new_df_columns = zip(df.columns[si:ei][positions_with_mutations], df.iloc[0,si:ei][positions_with_mutations])
    new_df_columns =["%s%s" % (b,a) for a,b in new_df_columns]     
    new_df = pd.concat([df["name"],df["seq"],df["activity"], 
                       df["num_muts"],df["p1"],df["p2"],
                       df.iloc[:,si:ei].iloc[:,positions_with_mutations]], axis=1)
    new_df.columns = ['name', 'seq', 'activity', 'num_muts', 'p1', 'p2'] + new_df_columns
    one_hot = get_one_hot_encoding(new_df, "Y20", "F253")
    si_oh = np.where(new_df.columns == "Y20")[0][0]
    ei_oh = np.where(new_df.columns == "F253")[0][0]
    assert one_hot.shape[1] == sum([len(pd.unique(new_df[C])) for C in new_df.columns[si_oh:(ei_oh+1)]])
    return new_df, embedding_paths, one_hot

def get_gcn4_df_and_embedding_paths():
    df = pd.read_csv(f"{base_path}/data/gcn4/gcn4.csv")
    embedding_paths = {
        "progen2-small": f"{base_path}/data/gcn4/embeddings/progen2-small",
        "progen2-medium": f"{base_path}/data/gcn4/embeddings/progen2-medium",
        "esm_35m": f"{base_path}/data/gcn4/embeddings/esm_35m",
        "esm_8m": f"{base_path}/data/gcn4/embeddings/esm_8m",
        "esm_3b": f"{base_path}/data/gcn4/embeddings/esm_3b",
        "esm_650m": f"{base_path}/data/gcn4/embeddings/esm_650m",
        "prot_bert": f"{base_path}/data/gcn4/embeddings/prot_bert",
        "esm_150m": f"{base_path}/data/gcn4/embeddings/esm_150m",
    }
    # The columns for one-hot
    first_col = "S101"
    last_col = "S144"
    one_hot = get_one_hot_encoding(df, first_col, last_col)
    si = np.where(df.columns == first_col)[0][0]
    ei = np.where(df.columns == last_col)[0][0]
    assert one_hot.shape[1] == sum([len(pd.unique(df[C])) for C in df.columns[si:(ei+1)]])
    return df, embedding_paths, one_hot, si, ei

mlp_base_parameters = {
    "activation": 'relu',           
    "solver": 'lbfgs', 
    "batch_size": 128,   
    "alpha": 1,                
    "learning_rate_init": 1e-3,    
    "max_iter": 50,
    "random_state": 4321,                
    #"early_stopping": False,         
    "n_iter_no_change": 10,         
    "verbose": False
}

one_hot_mlp_base_parameters = {
            "activation" : 'relu',           
            "solver" : 'adam', 
            "batch_size": 30,   
            "alpha" : 1e-4,                
            "learning_rate_init" : 2e-4,    
            "max_iter" :100,
            "random_state" : 4321,                
            #"early_stopping" : True,         
            "n_iter_no_change" : 10, 
            #"random_state" : 42,
            "verbose": False    
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--indices_pickle_file', type=str, required=True, help='Path to the indices pickle file')
    parser.add_argument('--log_labels', action='store_true', default=False, help='Flag to log the labels')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Flag to overwrite existing results file')
    parser.add_argument('--dataset', type=str, default='nmt', choices=['nmt', 'gcn4'], help='Choose nmt or gcn4')

    args = parser.parse_args()
    print("##### USING INDICES PICKLE FILE: %s" % args.indices_pickle_file)
    print("##### USING MODEL NAME: %s" % args.model_name)
    print("##### DATASET: %s" % args.dataset)

    if args.dataset == 'nmt':
        new_df, embedding_paths, one_hot = load_nmt()
        indices_col = 'indices.pt'
        labels_col = 'y_values.pt'
        embeddings_col = 'embeddings.pt'
        # For nmt, infer embedding keys and so on
    elif args.dataset == 'gcn4':
        new_df, embedding_paths, one_hot, si, ei = get_gcn4_df_and_embedding_paths()
        indices_col = 'indices.pt'
        labels_col = 'y_values.pt'
        embeddings_col = 'embeddings.pt'
    else:
        raise ValueError("Dataset must be 'nmt' or 'gcn4'")

    labels_all = {}
    indices_all = {}
    embeddings_all = {}

    print("Loading embeddings...")
    for model_name, model_path in embedding_paths.items():
        labels = torch.load(os.path.join(model_path, labels_col))
        indices = torch.load(os.path.join(model_path, indices_col))
        embeddings = torch.load(os.path.join(model_path, embeddings_col))

        flat_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        normalized_embeddings = flat_embeddings - flat_embeddings.mean(dim=0, keepdim=True)
        normalized_embeddings = normalized_embeddings / flat_embeddings.std(dim=0, keepdim=True)
        labels_all[model_name] = labels
        indices_all[model_name] = indices
        embeddings_all[model_name] = normalized_embeddings
    embeddings_all["one_hot"] = one_hot
    if args.model_name not in labels_all:
        # fallback
        labels_all["one_hot"] = labels
        indices_all["one_hot"] = indices
    else:
        labels = labels_all[args.model_name]
        indices = indices_all[args.model_name]
        # Only used if model_name not found? Used in selection for one_hot.

    results_df = pd.DataFrame()

    if args.dataset == "gcn4":
        result_dir = f"{base_path}/data/gcn4/results/by_complexity_and_budget_up_to_8"
    else: # nmt
        result_dir = f"{base_path}/data/nmt/results/by_complexity_and_budget_up_to_11"
    os.makedirs(result_dir, exist_ok=True)
    file_path = os.path.join(result_dir, f"mlp_llm_200_20_{args.model_name}")

    try:
        existing_df = pd.read_csv(file_path)
        print("Successfully read file:", file_path)
        results_df = existing_df
    except Exception as e:
        print("Could not read the file:", file_path)
        print("Reason:", e)

    all_sets = pickle.load(open(args.indices_pickle_file, "rb"))
    labels = labels_all[args.model_name].numpy()

    if args.log_labels:
        print("##### LOGGING LABELS")
        print("BEFORE LOGGING: %s" % labels)
        labels = np.log(labels)
        labels[np.isinf(labels)] = 0
        print("AFTER LOGGING: %s" % labels)
    embeddings = embeddings_all[args.model_name]
    for set_name in all_sets.keys():
        held_out_indices = all_sets[set_name]["held_out_indices"]
        sampled_by_budget_and_complexity = all_sets[set_name]["sampled_by_budget_and_complexity"]

        print("########################################################")
        print("SET NAME %s" % set_name)
        print("########################################################")
        if len(held_out_indices) > 4:
            print("held out indices: %d %d %d %d .... [%d sequences]" % 
                 tuple([held_out_indices[i] for i in range(4)] + [len(held_out_indices)]))

        for sampling_regime in sampled_by_budget_and_complexity.keys():
            max_muts_in_train = int(sampling_regime.split("up_to_")[1])
            print("Sampling regime: '%s' [Maximum %d mutations in train]" % (sampling_regime, max_muts_in_train))

            for budget_name in sampled_by_budget_and_complexity[sampling_regime].keys():
                budget = int(budget_name.split("budget_")[1])
                all_sampled_indices_in_regime_budget = sampled_by_budget_and_complexity[sampling_regime][budget_name]

                print("Budget:  '%s' [%d sequences], %d Iterations for budget" % (budget_name, budget, len(all_sampled_indices_in_regime_budget)))
                
                for idx in range(len(all_sampled_indices_in_regime_budget)):
                    train_sample_indices = all_sampled_indices_in_regime_budget[idx]
                    if idx == 0:
                        print("\t\tIteration %d / %d -> [%d sequences]" % (idx, len(all_sampled_indices_in_regime_budget), len(train_sample_indices)))
                        print("\t\t%s" % str(train_sample_indices))
                        print("\t\t%s" % str(new_df["num_muts"].iloc[train_sample_indices].to_list()))

                    is_exist = False
                    if not results_df.empty:
                        exists_mask = (
                            (results_df["set_name"] == set_name) &
                            (results_df["sampling_regime"] == sampling_regime) &
                            (results_df["max_muts_in_train"] == max_muts_in_train) &
                            (results_df["budget_name"] == budget_name) &
                            (results_df["budget"] == budget) &
                            (results_df["idx"] == idx)
                        ) 
                        is_exist = exists_mask.any()
                    if not args.overwrite and is_exist:
                        print("\t\t\tSkipping idx=%d in budget=%s and regime=%s, already exists in results_df." % (idx, budget_name, sampling_regime))
                    else:
                        if args.model_name == "one_hot":
                            mlp = MLPRegressor(hidden_layer_sizes=(500,50,), **one_hot_mlp_base_parameters)
                        else:
                            mlp = MLPRegressor(hidden_layer_sizes=(200,20,), **mlp_base_parameters)
                        mlp.fit(embeddings.numpy()[train_sample_indices], labels[train_sample_indices])
                        cor_llm = spearmanr(mlp.predict(embeddings.numpy()[held_out_indices]), labels[held_out_indices])
                        print("\t\t\tSpearman correlation: %f" % cor_llm.correlation)
                        iter_results_df = pd.DataFrame([{
                            "set_name": set_name,
                            "sampling_regime": sampling_regime,
                            "max_muts_in_train": max_muts_in_train,
                            "budget_name": budget_name,
                            "budget": budget,
                            "idx": idx,
                            "train_sample_indices": [list(train_sample_indices)],
                            "num_muts": [new_df["num_muts"].iloc[train_sample_indices].to_list()],
                            "cor_llm": cor_llm.correlation
                        }])
                        results_df = pd.concat([results_df, iter_results_df], ignore_index=True)
                        results_df.to_csv(file_path, index=False)

if __name__ == '__main__':
    main()
