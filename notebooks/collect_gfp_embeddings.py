import pandas as pd
import numpy as np
import os
import torch
import argparse

parser = argparse.ArgumentParser(description="Collect GFP embeddings and organize/split by mutation counts.")
parser.add_argument("--input_csv", type=str, default="data/gfp/gfp_dataset_10mut.csv", help="Path to the input CSV file.")
parser.add_argument("--model_name", type=str, default="esm_650m", help="Name of the model.")
parser.add_argument("--hidden_dim_size", type=int, default=1280, help="Hidden dimension size.")
parser.add_argument("--positions_to_select", type=int, nargs="+", default=None, help="Model-token positions to select from full-position embeddings.")
parser.add_argument("--position_indexing", choices=["model", "embedding"], default="model", help="'model' means positions include the BOS offset, matching embedding_engine --positions_to_embed. 'embedding' means raw zero-based embedding tensor indices.")
parser.add_argument("--mutations_to_select", type=int, nargs="+", default=None, help="num_muts values to save. Defaults to 0 through 10.")
parser.add_argument("--num_muts_colname", type=str, default="num_muts", help="Column name containing the number of mutations.")


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


def select_positions(embeddings, positions_to_select, position_indexing):
    if positions_to_select is None:
        return embeddings

    if position_indexing == "model":
        bad_positions = [pos for pos in positions_to_select if pos < 1]
        if bad_positions:
            raise ValueError("Model-token positions must be >= 1: %s" % bad_positions)
        position_indices = [pos - 1 for pos in positions_to_select]
    else:
        bad_positions = [pos for pos in positions_to_select if pos < 0]
        if bad_positions:
            raise ValueError("Embedding tensor indices must be >= 0: %s" % bad_positions)
        position_indices = positions_to_select

    if max(position_indices) >= embeddings.shape[1]:
        raise IndexError(
            "Requested position index %d, but embeddings only have %d positions. "
            "If these embeddings were already generated with --positions_to_embed, "
            "do not pass --positions_to_select to the collector."
            % (max(position_indices), embeddings.shape[1])
        )

    return embeddings[:, torch.tensor(position_indices), :]


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
mutations_to_select = args.mutations_to_select if args.mutations_to_select is not None else list(range(0, 11))

if args.num_muts_colname not in df.columns:
    raise ValueError("Column '%s' not found in %s" % (args.num_muts_colname, args.input_csv))

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

    embedding_all = None
    label_all = torch.zeros([df.shape[0]], dtype=torch.float)
    indices_all = torch.zeros([df.shape[0]], dtype=torch.int64)

    for i, train_dir in enumerate(train_dirs):

        subfolder = os.path.relpath(os.path.dirname(train_dir), intermediate_result_path)
        print("\tLoading %s [%d/%d]" % (subfolder, i, len(train_dirs)))
        embeddings = load_tensor(os.path.join(train_dir, "embeddings.pt"))
        embeddings = select_positions(embeddings, args.positions_to_select, args.position_indexing)

        if embeddings.shape[2] != hs:
            raise ValueError("Expected hidden_dim_size=%d, but loaded embeddings have hidden dim %d" % (hs, embeddings.shape[2]))

        if embedding_all is None:
            embedding_all = torch.zeros([df.shape[0], embeddings.shape[1], hs], dtype=torch.float)

        print("MEAN: ", embeddings.mean(dim=1).mean(dim=1).mean())
        print("STD: ", embeddings.mean(dim=1).mean(dim=1).std())

        labels = load_tensor(os.path.join(train_dir, "y_value.pt"))
        indices = load_tensor(os.path.join(train_dir, "indices.pt"))
        
        indices_all[indices] = indices
        label_all[indices] = labels.to(torch.float)
        embedding_all[indices] = embeddings

    for progress_i, i in enumerate(mutations_to_select):
        slice_indices = np.where(df[args.num_muts_colname] == i)[0]
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "embeddings_of_nmut_%d.pt" % i), progress_i, len(mutations_to_select) - 1))
        torch.save(embedding_all[slice_indices], os.path.join(final_save_path, "embeddings_of_nmut_%d.pt" % i))
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "y_values_of_nmut_%d.pt" % i), progress_i, len(mutations_to_select) - 1))
        torch.save(label_all[slice_indices], os.path.join(final_save_path, "y_values_of_nmut_%d.pt" % i))
        print("\tSaving %s [%d/%d]" % (os.path.join(final_save_path, "indices_of_nmut_%d.pt" % i), progress_i, len(mutations_to_select) - 1))
        torch.save(indices_all[slice_indices], os.path.join(final_save_path, "indices_of_nmut_%d.pt" % i))
    
    del embedding_all, label_all, indices_all
