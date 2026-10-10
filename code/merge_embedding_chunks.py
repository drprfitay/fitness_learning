import argparse
import os
import re

import pandas as pd
import torch


def load_pt(path):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def nmut_from_name(path):
    match = re.search(r"_of_nmuts?_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def chunk_paths(folder, prefix):
    paths = []
    for name in os.listdir(folder):
        if name.startswith(prefix + "_of_nmut") and name.endswith(".pt"):
            paths.append(os.path.join(folder, name))
    return sorted(paths, key=nmut_from_name)


def merge_one_folder(folder, df=None, y_col=None, max_nmut=None, num_muts_col="num_muts", skip_y_validation=False):
    emb_paths = chunk_paths(folder, "embeddings")
    if max_nmut is not None:
        emb_paths = [path for path in emb_paths if nmut_from_name(path) <= max_nmut]
    if len(emb_paths) == 0:
        raise FileNotFoundError("No embeddings_of_nmut_*.pt files in %s" % folder)

    all_embeddings = []
    all_indices = []
    all_y = []

    for emb_path in emb_paths:
        nmut = nmut_from_name(emb_path)
        idx_path = os.path.join(folder, "indices_of_nmut_%d.pt" % nmut)
        y_path = os.path.join(folder, "y_values_of_nmut_%d.pt" % nmut)

        if not os.path.exists(idx_path):
            idx_path = os.path.join(folder, "indices_of_nmuts_%d.pt" % nmut)
        if not os.path.exists(y_path):
            y_path = os.path.join(folder, "y_values_of_nmuts_%d.pt" % nmut)
        if not os.path.exists(idx_path):
            raise FileNotFoundError("Missing index chunk for nmut=%d in %s" % (nmut, folder))
        if not os.path.exists(y_path):
            raise FileNotFoundError("Missing y chunk for nmut=%d in %s" % (nmut, folder))

        embeddings = load_pt(emb_path)
        indices = load_pt(idx_path).long().view(-1)
        y_values = load_pt(y_path).view(-1)

        if embeddings.shape[0] != indices.numel():
            raise ValueError("Chunk length mismatch for nmut=%d in %s" % (nmut, folder))
        if indices.numel() != y_values.numel():
            raise ValueError("Chunk y length mismatch for nmut=%d in %s" % (nmut, folder))

        all_embeddings.append(embeddings)
        all_indices.append(indices)
        all_y.append(y_values)

    embeddings = torch.cat(all_embeddings, dim=0)
    indices = torch.cat(all_indices, dim=0)
    y_values = torch.cat(all_y, dim=0)

    order = torch.argsort(indices)
    embeddings = embeddings[order]
    indices = indices[order]
    y_values = y_values[order]

    unique_indices = torch.unique(indices)
    if unique_indices.numel() != indices.numel():
        raise ValueError("Duplicate indices found in %s" % folder)

    if df is not None:
        expected_indices = torch.arange(len(df))
        if max_nmut is not None:
            if num_muts_col not in df.columns:
                raise ValueError("Cannot validate --max_nmut because df is missing column %r" % num_muts_col)
            expected_indices = torch.as_tensor(
                df.index[df[num_muts_col].astype(int) <= int(max_nmut)].to_numpy(),
                dtype=indices.dtype,
            )
        if indices.numel() != expected_indices.numel():
            raise ValueError(
                "Merged rows in %s are %d, expected %d from dataframe"
                % (folder, indices.numel(), expected_indices.numel())
            )
        if indices.max().item() >= len(df):
            raise ValueError("Index outside dataframe length in %s" % folder)
        if not torch.equal(indices.cpu(), expected_indices.cpu()):
            raise ValueError("Merged indices in %s do not cover expected dataframe rows exactly" % folder)
        if y_col is not None and not skip_y_validation:
            expected_y = torch.as_tensor(df.iloc[indices.numpy()][y_col].to_numpy(), dtype=y_values.dtype)
            if not torch.allclose(y_values.cpu(), expected_y.cpu(), equal_nan=True):
                close = torch.isclose(y_values.cpu(), expected_y.cpu(), equal_nan=True)
                bad = torch.where(~close)[0]
                diff = y_values.cpu() - expected_y.cpu()
                print("y_values mismatch in %s" % folder)
                print("bad rows: %d / %d" % (bad.numel(), y_values.numel()))
                print("max abs diff: %s" % diff[bad].abs().max().item())
                for i in bad[:10].tolist():
                    print(
                        "  row=%d index=%d saved=%s expected=%s diff=%s" %
                        (i, indices[i].item(), y_values[i].item(), expected_y[i].item(), diff[i].item())
                    )
                raise ValueError("y_values do not match df[%r] in %s" % (y_col, folder))

    torch.save(embeddings, os.path.join(folder, "embeddings.pt"))
    torch.save(y_values, os.path.join(folder, "y_values.pt"))
    torch.save(indices, os.path.join(folder, "indices.pt"))

    print("%s: saved %d rows ordered by indices" % (folder, indices.numel()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--dataset_csv")
    parser.add_argument("--y_col")
    parser.add_argument(
        "--skip_y_validation",
        "--skip_y_values",
        dest="skip_y_validation",
        action="store_true",
        default=False,
    )
    parser.add_argument("--max_nmut", "--max_nmuts", "--max_muts", type=int, default=None)
    parser.add_argument("--num_muts_col", default="num_muts")
    parser.add_argument("--embedding_keys", nargs="*", default=[])
    parser.add_argument("--embedding_dirs", nargs="*", default=[])
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    dataset_csv = args.dataset_csv
    if dataset_csv is None and args.dataset:
        candidate = os.path.join(here, "data", args.dataset, "%s.csv" % args.dataset)
        dataset_csv = candidate if os.path.exists(candidate) else None

    df = pd.read_csv(dataset_csv) if dataset_csv else None
    folders = list(args.embedding_dirs)

    if args.dataset and args.embedding_keys:
        folders += [
            os.path.join(here, "data", args.dataset, "embeddings", key)
            for key in args.embedding_keys
        ]

    if len(folders) == 0:
        raise ValueError("Provide --embedding_dirs or --dataset with --embedding_keys")

    for folder in folders:
        merge_one_folder(
            folder,
            df=df,
            y_col=args.y_col,
            max_nmut=args.max_nmut,
            num_muts_col=args.num_muts_col,
            skip_y_validation=args.skip_y_validation,
        )


if __name__ == "__main__":
    main()
