import pandas as pd
import numpy as np
import os
import torch
import argparse

parser = argparse.ArgumentParser(description="Collect GFP embeddings and organize/split by mutation counts.")
parser.add_argument("--input_csv", type=str, default="data/gfp/gfp_dataset_10mut.csv", help="Path to the input CSV file.")
parser.add_argument("--model_name", type=str, default="esm_650m", help="Name of the model.")
parser.add_argument("--hidden_dim_size", type=int, default=1280, help="Hidden dimension size.")


args = parser.parse_args()

df = pd.read_csv(args.input_csv)

REQUIRED_TRAIN_FILES = ("embeddings.pt", "y_value.pt", "indices.pt")


def load_tensor(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def find_train_dirs(intermediate_result_path):
    train_dirs = []
    for root, _, _ in os.walk(intermediate_result_path):
        if os.path.basename(root) != "train":
            continue
        if all(os.path.exists(os.path.join(root, filename)) for filename in REQUIRED_TRAIN_FILES):
            train_dirs.append(root)
    return sorted(train_dirs)


base_path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/"

# intermediate_result_paths = [#"%s/results/gfp_embeddings/esm_650m/evaluations" % base_path,
#                             "%s/results/gfp_embeddings/esm_35m/evaluations" % base_path,
#                             "%s/results/gfp_embeddings/esm_8m/evaluations" % base_path]

# final_save_path = [#"%s/data/gfp/embeddings/esm_650m" % base_path,
#                     "%s/data/gfp/embeddings/esm_35m" % base_path,
#                     "%s/data/gfp/embeddings/esm_8m" % base_path]


intermediate_result_paths = ["%s/results/gfp_embeddings/%s/evaluations" % (base_path, args.model_name)]
final_save_path = ["%s/data/gfp/new_embeddings/%s" % (base_path, args.model_name)]
hidden_dim_size = [args.hidden_dim_size]

# hidden_dim_size = [#1280, 
# 480, 320]


for intermediate_result_path, final_save_path, hs in zip(intermediate_result_paths, final_save_path, hidden_dim_size):

    os.makedirs(final_save_path, exist_ok=True)
    
    print("Processing %s and saving to %s" % (intermediate_result_path, final_save_path))

    train_dirs = find_train_dirs(intermediate_result_path)
    if not train_dirs:
        raise FileNotFoundError(
            "No train folders containing %s found under %s"
            % (", ".join(REQUIRED_TRAIN_FILES), intermediate_result_path)
        )

    embedding_all = torch.zeros([df.shape[0], 22, hs], dtype=torch.float)
    label_all = torch.zeros([df.shape[0]], dtype=torch.float)
    indices_all = torch.zeros([df.shape[0]], dtype=torch.int64)

    for i, train_dir in enumerate(train_dirs):

        subfolder = os.path.relpath(os.path.dirname(train_dir), intermediate_result_path)
        print("\tLoading %s [%d/%d]" % (subfolder, i, len(train_dirs)))
        embeddings = load_tensor(os.path.join(train_dir, "embeddings.pt"))

        print("MEAN: ", embeddings.mean(dim=1).mean(dim=1).mean())
        print("STD: ", embeddings.mean(dim=1).mean(dim=1).std())

        labels = load_tensor(os.path.join(train_dir, "y_value.pt"))
        indices = load_tensor(os.path.join(train_dir, "indices.pt"))
        
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
