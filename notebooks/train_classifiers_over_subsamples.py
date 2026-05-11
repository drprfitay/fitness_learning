import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt

def evaluate_classifier(score, 
                        predicted_label, 
                        gt_label,
                        label_true=0,
                        label_false=1):

    tp = sum((predicted_label == label_true) & (gt_label == label_true))
    tn = sum((predicted_label == label_false) & (gt_label == label_false))
    fp = sum((predicted_label == label_true) & (gt_label == label_false))
    fn = sum((predicted_label == label_false) & (gt_label == label_true))
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = np.sum(predicted_label == gt_label) / len(predicted_label)
    roc = roc_auc_score(gt_label, score)

    ordered_scores = np.argsort(score)[0:100]
    top_100_pct = sum(gt_label[ordered_scores] == label_true) / 100

    evaluation = {
        "tp" : tp,
        "tn" : tn,
        "fp" : fp,
        "fn" : fn,
        "precision" : precision,
        "recall" : recall,
        "f1" : f1,
        "accuracy" : accuracy,
        "roc" : roc,
        "top_100_pct": top_100_pct
    }
    #roc = sklearn.
    return evaluation
    


# aggregate all the data

base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

def get_one_hot_encoding(sdf, first_col, last_col):
    si = np.where(sdf.columns == first_col)[0][0]
    ei = np.where(sdf.columns == last_col)[0][0]
    
    one_hot_encoding = torch.from_numpy(pd.get_dummies(sdf[sdf.columns[si:(ei+1)]]).to_numpy()).to(torch.int64)

    return(one_hot_encoding)
    
df = pd.read_csv("data/gfp/gfp_dataset_10mut.csv")
one_hot = get_one_hot_encoding(df, "L42", "V224").numpy()
si = np.where(df.columns == "L42")[0][0]
ei = np.where(df.columns == "V224")[0][0]
assert one_hot.shape[1] == sum([len(pd.unique(df[C])) for C in df.columns[si:(ei+1)]])


base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

embedding_paths = ["%s/data/gfp/embeddings/esm_650m" % base_path,
                    "%s/data/gfp/embeddings/esm_35m" % base_path,
                    "%s/data/gfp/embeddings/esm_8m" % base_path]


classifier_embeddings_path = embedding_paths[1]


#for fraction in [0.005, 0.01, 0.0125, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
for sample_size in [10, 25, 50, 75, 100, 250, 500, 1000, 25000, 5000, 10000, 25000, 50000]:
    across_iterations_llm = []
    across_iterations_ohe = []

    #use_fraction = fraction
    for iter in range(20):
        
        print("Iteration %d" % iter)
        saved_train_parameters = {}
            
        for n_train in range(1, 10):

            if n_train > 8:
                use_fraction = 0

            ohe_labels = df["inactive"].astype(int).to_numpy()

            train_indices = np.where((df["num_muts"] == n_train).to_numpy())[0]
            sub_df = df.iloc[train_indices]
            sub_ohe_labels = ohe_labels[train_indices]
            sub_one_hot = one_hot[train_indices]
            labels = torch.load(os.path.join(classifier_embeddings_path, "y_values_of_nmut_%d.pt" % n_train))
            indices = torch.load(os.path.join(classifier_embeddings_path, "indices_of_nmut_%d.pt" % n_train))
            embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_%d.pt" % n_train))
            
            # sanity
            assert(sum(labels == sub_ohe_labels) == len(sub_ohe_labels))
            assert(sum(indices == train_indices) == len(train_indices))


            #n_samples = max(1,int(len(train_indices) * use_fraction))
            n_samples = min(sample_size, len(train_indices)) 
            true_indices = np.where(train_indices)[0]
            sampled_indices = np.random.choice(true_indices, size=n_samples, replace=False)
            subsample_vector = np.zeros_like(train_indices, dtype=bool)
            subsample_vector[sampled_indices] = True
            

            print("\t Sampled %d sequences with %d mutations" % (n_samples, n_train))

            saved_train_parameters[n_train] =\
                {"train_indices": subsample_vector,
                "train_labels": sub_ohe_labels[sampled_indices],
                "train_embeddings": embeddings.numpy()[sampled_indices],
                "train_ohe": sub_one_hot[sampled_indices]}
        
        embeddings = []
        ohe = []
        labels = []

        for k, train_dict in saved_train_parameters.items():
            print("Aggregating parameters % s" % str(k))

            embeddings.append(train_dict["train_embeddings"])
            ohe.append(train_dict["train_ohe"])
            labels.append(train_dict["train_labels"])

        embeddings = np.concatenate(embeddings, axis=0)
        ohe = np.concatenate(ohe, axis=0)
        labels = np.concatenate(labels, axis=0)

        print(saved_train_parameters)

        flat_embeddings = embeddings.reshape(embeddings.shape[0], -1)
        normalized_embeddings = flat_embeddings - flat_embeddings.mean(axis=0, keepdims=True)
        normalized_embeddings = normalized_embeddings / flat_embeddings.std(axis=0, keepdims=True)


        mlp_base_parameters = {
            "activation" : 'relu',           
            "solver" : 'lbfgs', 
            "batch_size": 128,   
            "alpha" : 1,                
            "learning_rate_init" : 1e-3,    
            "max_iter" :50,
            "random_state" : 4321,                
            "early_stopping" : True,         
            "n_iter_no_change" : 200,         
            "random_state" : 42,
            "verbose": True
        }


        print("Fitting MLP over %dx%d embeddings to %d labels" % (normalized_embeddings.shape[0], normalized_embeddings.shape[1], len((labels))))

        mlp_llm = MLPClassifier(hidden_layer_sizes=(64,), **mlp_base_parameters)
        mlp_llm.fit(normalized_embeddings, labels)

        print("Fitting MLP over %dx%d ohe to %d labels" % (ohe.shape[0], ohe.shape[1], len((labels))))

        mlp_ohe = MLPClassifier(hidden_layer_sizes=(200,), **mlp_base_parameters)
        mlp_ohe.fit(ohe, labels)


        score_llm = []
        labels_llm = []
        predicted_labels_llm = []

        score_ohe =[]
        labels_ohe = []
        predicted_labels_ohe = []
        
        # evaluation loop
        for n_train in range(1, 10):

            ohe_labels = df["inactive"].astype(int).to_numpy()

            train_indices = np.where((df["num_muts"] == n_train).to_numpy())[0]
            sub_df = df.iloc[train_indices]
            sub_ohe_labels = ohe_labels[train_indices]
            sub_one_hot = one_hot[train_indices]

            labels = torch.load(os.path.join(classifier_embeddings_path, "y_values_of_nmut_%d.pt" % n_train))
            indices = torch.load(os.path.join(classifier_embeddings_path, "indices_of_nmut_%d.pt" % n_train))
            embeddings = torch.load(os.path.join(classifier_embeddings_path, "embeddings_of_nmut_%d.pt" % n_train))
            
            # sanity
            assert(sum(labels == sub_ohe_labels) == len(sub_ohe_labels))
            assert(sum(indices == train_indices) == len(train_indices))

            test_indices = ~saved_train_parameters[n_train]["train_indices"]

            if sum(test_indices) == 0:
                continue

            flat_embeddings = embeddings.reshape(embeddings.shape[0], -1)
            normalized_embeddings = flat_embeddings - flat_embeddings.mean(axis=0, keepdims=True)
            normalized_embeddings = normalized_embeddings / flat_embeddings.std(axis=0, keepdims=True)

            print("Evauating llm over %d test_sequences with %d mutations" % (sum(test_indices), n_train))
            predictions_proba = mlp_llm.predict_proba(normalized_embeddings[test_indices].numpy())
            predictions = (predictions_proba[:, 1] > 0.5).astype(int)

        
            score_llm.append(predictions_proba[:, 1])
            labels_llm.append(labels[test_indices])
            predicted_labels_llm.append(predictions)

            print("Evauating OHE over %d test_sequences with %d mutations" % (sum(test_indices), n_train))
            print(sub_one_hot[test_indices].shape)
            print(test_indices.shape)

            predictions_proba = mlp_ohe.predict_proba(sub_one_hot[test_indices])
            predictions = (predictions_proba[:, 1] > 0.5).astype(int)

            score_ohe.append(predictions_proba[:, 1])
            labels_ohe.append(labels[test_indices])
            predicted_labels_ohe.append(predictions)

        all_scores_llm = np.concatenate(score_llm, axis=0)
        all_labels_llm = np.concatenate(labels_llm, axis=0)
        all_predicted_labels_llm = np.concatenate(predicted_labels_llm, axis=0)

        all_scores_ohe = np.concatenate(score_ohe, axis=0)
        all_labels_ohe = np.concatenate(labels_ohe, axis=0)
        all_predicted_labels_ohe = np.concatenate(predicted_labels_ohe, axis=0)

        evaluation = evaluate_classifier(all_scores_llm, all_predicted_labels_llm, all_labels_llm)
        print("LLM evaluation: %s" % str(evaluation))

        evaluation["fraction"] = sample_size
        evaluation["iter"] = iter
        across_iterations_llm.append(evaluation)

        evaluation = evaluate_classifier(all_scores_ohe, all_predicted_labels_ohe, all_labels_ohe)
        print("OHE evaluation: %s" % str(evaluation))
        evaluation["fraction"] = sample_size
        evaluation["iter"] = iter
        across_iterations_ohe.append(evaluation)

        across_iterations_llm_df = pd.DataFrame(across_iterations_llm)
        across_iterations_ohe_df = pd.DataFrame(across_iterations_ohe)
        across_iterations_llm_df.to_csv("data/gfp/subsamples/s_across_iterations_llm_fraction_new_new_%d.csv" % sample_size, index=False)
        across_iterations_ohe_df.to_csv("data/gfp/subsamples/s_across_iterations_ohe_fraction_new_new_%d.csv" % sample_size, index=False)

