import pandas as pd
import numpy as np
import os
import torch
import argparse

parser = argparse.ArgumentParser(description="Collect embeddings and organize/split by mutation counts.")
parser.add_argument("--input_csv", type=str, default="", help="Path to the input CSV file.")
parser.add_argument("--model_name", type=str, default="esm_650m", help="Name of the model.")
parser.add_argument("--hidden_dim_size", type=int, default=1280, help="Hidden dimension size.")
parser.add_argument("--dataset_name", type=str, default="gfp", help="Name of the dataset.")


args = parser.parse_args()

print("Input CSV: ", args.input_csv)
print("Model name: ", args.model_name)
print("Hidden dimension size: ", args.hidden_dim_size)
print("Dataset name: ", args.dataset_name)

df = pd.read_csv(args.input_csv)


base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"
intermediate_result_paths = ["%s/results/%s/%s/evaluations" % (base_path, args.dataset_name, args.model_name)]
final_save_path = ["%s/data/%s/new_embeddings/%s" % (base_path, args.dataset_name, args.model_name)]
hidden_dim_size = [args.hidden_dim_size]

# hidden_dim_size = [#1280, 
# 480, 320]


for intermediate_result_path, final_save_path, hs in zip(intermediate_result_paths, final_save_path, hidden_dim_size):

    os.makedirs(final_save_path, exist_ok=True)
    
    print("Processing %s and saving to %s" % (intermediate_result_path, final_save_path))

    subfolders = [f.name for f in os.scandir(intermediate_result_path) if f.is_dir()]

    #embedding_all = torch.zeros([df.shape[0], 22, hs], dtype=torch.float)
    embedding_all = None
    label_all = torch.zeros([df.shape[0]], dtype=torch.float)
    indices_all = torch.zeros([df.shape[0]], dtype=torch.int64)

    for i, subfolder in enumerate(subfolders):

        print("\tLoading %s [%d/%d]" % (subfolder, i, len(subfolders)))
        embeddings = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "embeddings.pt"))

        if embedding_all is None:
            _, P, hs = embeddings.shape
            embedding_all = torch.zeros([df.shape[0], P, hs], dtype=torch.float)
            print("Creating embeddings all matrix with shape: ", embedding_all.shape)

        print("MEAN: ", embeddings.mean(dim=1).mean(dim=1).mean())
        print("STD: ", embeddings.mean(dim=1).mean(dim=1).std())

        labels = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "y_value.pt"))
        indices = torch.load(os.path.join(intermediate_result_path, subfolder, "train", "indices.pt"))
        
        indices_all[indices] = indices
        label_all[indices] = labels.to(torch.float)
        embedding_all[indices] = embeddings

    for i in range(0, 11):
        slice_indices = np.where(df["num_muts"] == i)[0]
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "embeddings_of_nmut_%d.pt" % i), i, 10))
        torch.save(embedding_all[slice_indices], os.path.join(final_save_path, "embeddings_of_nmut_%d.pt" % i))
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "y_values_of_nmut_%d.pt" % i), i, 10))
        torch.save(label_all[slice_indices], os.path.join(final_save_path, "y_values_of_nmut_%d.pt" % i))
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "indices_of_nmut_%d.pt" % i), i, 10))
        torch.save(indices_all[slice_indices], os.path.join(final_save_path, "indices_of_nmut_%d.pt" % i))
    
    del embedding_all, label_all, indices_all
