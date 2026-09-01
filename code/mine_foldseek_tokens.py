#!/usr/bin/env python3
"""Generate Foldseek 3Di structural tokens for SaProt datasets.

Notebook-style defaults:
- dataset CSVs use the same names/paths as notebooks/utils_for_analysis.py
- relative dataset paths are resolved from the notebooks/ directory
- default PDB path is the CSV path with .pdb instead of .csv
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
NOTEBOOKS_DIR = os.path.join(PROJECT_DIR, "notebooks")


# Edit these values when running the file directly, or call
# make_foldseek_tokens_for_dataset(...) from a notebook.
DATASET_NAME = "his2"
CSV_PATH = None
PDB_PATH = None
OUTPUT_PATH = None
SAVE_OUTPUT = True
PRINT_TOKENS = False
SAVE_TEXT_FILES = True


DATASET_PATHS = {
    "gfp": "data/gfp/gfp_dataset_10mut.csv",
    "lov": "data/lov/lov.csv",
    "pard3": "data/pard3/pard3.csv",
    "gcn4": "data/gcn4/gcn4.csv",
    "pte": "data/pte/pte.csv",
    "nmt": "data/nmt/nmt.csv",
    "aamyl": "data/aamyl/aamyl.csv",
    "trpb": "data/trpb/trpb.csv",
    "his": "data/his/his.csv",
    "his2": "data/his2/his2.csv",
    "his5": "data/his5/his5.csv",
    "casp": "data/casp/casp.csv",
}

FULL_SEQ_COLUMN_NAME = {
    "gcn4": "full_seq",
    "pard3": "full_seq",
    "lov": "full_seq",
    "gfp": "full_seq",
    "pte": "full_seq",
    "aamyl": "full_seq",
    "nmt": "full_seq",
    "trpb": "full_seq",
    "his": "full_seq",
    "his2": "full_seq",
    "his5": "full_seq",
    "casp": "full_seq",
}

NUM_MUTS_COLUMN_NAME = {
    "pard3": "num_muts",
    "lov": "num_muts",
    "gfp": "num_muts",
    "pte": "num_muts",
    "gcn4": "num_muts",
    "nmt": "num_muts",
    "aamyl": "num_muts",
    "trpb": "num_muts",
    "his": "num_muts",
    "his2": "num_muts",
    "his5": "num_muts",
    "casp": "num_muts",
}


def resolve_dataset_csv(dataset_name, csv_path=None):
    if csv_path is None:
        if dataset_name not in DATASET_PATHS:
            raise KeyError(
                "Unknown dataset %r. Available datasets: %s" %
                (dataset_name, ", ".join(sorted(DATASET_PATHS)))
            )
        csv_path = DATASET_PATHS[dataset_name]

    if not os.path.isabs(csv_path):
        csv_path = os.path.join(NOTEBOOKS_DIR, csv_path)
    return os.path.abspath(csv_path)


def resolve_pdb_path(csv_path, pdb_path=None):
    if pdb_path is None:
        base_path = os.path.splitext(csv_path)[0]
        candidate_paths = [
            base_path + ".pdb",
            base_path + ".cif",
            base_path + ".mmcif",
        ]
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return os.path.abspath(candidate_path)
        pdb_path = candidate_paths[0]
    elif not os.path.isabs(pdb_path):
        pdb_path = os.path.join(NOTEBOOKS_DIR, pdb_path)
    return os.path.abspath(pdb_path)


def get_dataset_columns(dataset_name, df, sequence_col=None, num_muts_col=None):
    if sequence_col is None:
        sequence_col = FULL_SEQ_COLUMN_NAME.get(dataset_name)
        if sequence_col is None:
            for candidate in ("full_seq", "seq", "full_sequence", "sequence"):
                if candidate in df.columns:
                    sequence_col = candidate
                    break

    if num_muts_col is None:
        num_muts_col = NUM_MUTS_COLUMN_NAME.get(dataset_name, "num_muts")

    if sequence_col not in df.columns:
        raise KeyError("Could not find sequence column %r in dataset" % sequence_col)
    if num_muts_col not in df.columns:
        raise KeyError("Could not find num_muts column %r in dataset" % num_muts_col)

    return sequence_col, num_muts_col


def read_foldseek_3di(pdb_path):
    stdout = ""
    stderr = ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tsv") as tmp:
        out_path = tmp.name

    try:
        completed = subprocess.run(
            ["foldseek", "structureto3didescriptor", pdb_path, out_path],
            check=True,
            capture_output=True,
            text=True,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError("foldseek executable was not found in PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        msg = stderr or stdout or "foldseek failed without output"
        raise RuntimeError("Foldseek failed for %s: %s" % (pdb_path, msg[-1000:])) from exc

    records = []
    with open(out_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                aa_seq = parts[1].upper()
                foldseek_3di = parts[2].lower()
                if len(aa_seq) != len(foldseek_3di):
                    raise ValueError(
                        "Foldseek sequence/token length mismatch for %s: %d != %d" %
                        (parts[0], len(aa_seq), len(foldseek_3di))
                    )
                records.append({
                    "name": parts[0],
                    "aa_seq": aa_seq,
                    "foldseek_3di": foldseek_3di,
                })

    os.remove(out_path)

    if completed.returncode != 0:
        raise RuntimeError("Foldseek failed for %s" % pdb_path)
    if len(records) == 0:
        msg = stderr or stdout or "Foldseek wrote no records and no diagnostic output"
        raise ValueError(
            "No Foldseek records found for %s. "
            "If this structure is mmCIF, pass the .cif/.mmcif path explicitly "
            "or keep it with a .cif/.mmcif suffix. Foldseek output: %s" %
            (pdb_path, msg[-1000:])
        )

    return records


def align_3di_to_wt(wt_seq, pdb_seq, foldseek_3di):
    """Align PDB-coordinate 3Di tokens to WT sequence like StructurePlmEmbedding."""
    wt_seq = wt_seq.upper()
    pdb_seq = pdb_seq.upper()
    foldseek_3di = foldseek_3di.lower()

    n = len(wt_seq)
    m = len(pdb_seq)

    match_score = 2
    mismatch_score = -1
    gap_score = -2

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]

    for j in range(1, m + 1):
        trace[0][j] = "pdb"
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_score
        trace[i][0] = "wt"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = dp[i - 1][j - 1] + (
                match_score if wt_seq[i - 1] == pdb_seq[j - 1] else mismatch_score
            )
            wt_gap = dp[i - 1][j] + gap_score
            pdb_gap = dp[i][j - 1] + gap_score

            best = max(diagonal, wt_gap, pdb_gap)
            dp[i][j] = best
            if diagonal == best:
                trace[i][j] = "diag"
            elif wt_gap == best:
                trace[i][j] = "wt"
            else:
                trace[i][j] = "pdb"

    j = max(range(m + 1), key=lambda x: dp[n][x])
    i = n
    mapping = [None] * n

    while i > 0:
        move = trace[i][j]
        if move == "diag":
            mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == "wt":
            i -= 1
        elif move == "pdb":
            j -= 1
        else:
            raise RuntimeError("Failed to trace WT/PDB alignment")

    aligned_3di = "".join(
        foldseek_3di[pdb_i] if pdb_i is not None else "#"
        for pdb_i in mapping
    )
    aligned_pdb = "".join(
        pdb_seq[pdb_i] if pdb_i is not None else "-"
        for pdb_i in mapping
    )

    covered = np.array([pdb_i is not None for pdb_i in mapping], dtype=bool)
    if covered.any():
        wt_array = np.array(list(wt_seq))
        pdb_array = np.array(list(aligned_pdb))
        identity = np.mean(wt_array[covered] == pdb_array[covered])
    else:
        identity = 0.0
    coverage = covered.mean() if len(covered) else 0.0

    return aligned_3di, aligned_pdb, mapping, float(identity), float(coverage)


def make_saprot_sequence(sequence, aligned_3di_to_wt):
    sequence = sequence.upper()
    if len(sequence) != len(aligned_3di_to_wt):
        raise ValueError(
            "Sequence and aligned 3Di token lengths differ: %d != %d" %
            (len(sequence), len(aligned_3di_to_wt))
        )
    return "".join(aa + token for aa, token in zip(sequence, aligned_3di_to_wt))


def make_foldseek_tokens_for_dataset(
    dataset_name,
    csv_path=None,
    pdb_path=None,
    output_path=None,
    save_output=True,
    save_text_files=True,
    print_tokens=False,
    sequence_col=None,
    num_muts_col=None,
):
    csv_path = resolve_dataset_csv(dataset_name, csv_path=csv_path)
    pdb_path = resolve_pdb_path(csv_path, pdb_path=pdb_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError("Could not find CSV: %s" % csv_path)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError("Could not find PDB: %s" % pdb_path)

    df = pd.read_csv(csv_path)
    sequence_col, num_muts_col = get_dataset_columns(
        dataset_name,
        df,
        sequence_col=sequence_col,
        num_muts_col=num_muts_col,
    )

    wt_idx = np.where(df[num_muts_col].to_numpy() == 0)[0]
    if len(wt_idx) != 1:
        raise ValueError("Expected exactly one WT row, found %d" % len(wt_idx))

    wt_idx = int(wt_idx[0])
    wt_seq = str(df[sequence_col].iloc[wt_idx]).upper()

    records = read_foldseek_3di(pdb_path)

    best = None
    for rec in records:
        aligned_3di, aligned_pdb, mapping, identity, coverage = align_3di_to_wt(
            wt_seq=wt_seq,
            pdb_seq=rec["aa_seq"],
            foldseek_3di=rec["foldseek_3di"],
        )
        score = identity * coverage

        if best is None or score > best["score"]:
            best = {
                **rec,
                "aligned_3di_to_wt": aligned_3di,
                "aligned_pdb_seq_to_wt": aligned_pdb,
                "structure_mapping": mapping,
                "identity_to_wt": identity,
                "coverage_to_wt": coverage,
                "score": score,
            }

    saprot_wt_sequence = make_saprot_sequence(wt_seq, best["aligned_3di_to_wt"])

    out = {
        "dataset": dataset_name,
        "csv_path": csv_path,
        "pdb_path": pdb_path,
        "sequence_col": sequence_col,
        "num_muts_col": num_muts_col,
        "wt_idx": wt_idx,
        "wt_seq": wt_seq,
        "saprot_wt_sequence": saprot_wt_sequence,
        **best,
    }

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(csv_path),
            "%s_foldseek_aligned_tokens.pkl" % dataset_name,
        )
    elif not os.path.isabs(output_path):
        output_path = os.path.join(NOTEBOOKS_DIR, output_path)

    output_dir = os.path.dirname(output_path)
    pdb_sequence_txt_path = os.path.join(output_dir, "%s_pdb_sequence.txt" % dataset_name)
    foldseek_token_sequence_txt_path = os.path.join(output_dir, "%s_foldseek_3di.txt" % dataset_name)

    out["pdb_sequence_txt_path"] = pdb_sequence_txt_path
    out["foldseek_token_sequence_txt_path"] = foldseek_token_sequence_txt_path

    if save_output:
        with open(output_path, "wb") as handle:
            pickle.dump(out, handle)

    if save_text_files:
        with open(pdb_sequence_txt_path, "w") as handle:
            handle.write(best["aa_seq"])
        with open(foldseek_token_sequence_txt_path, "w") as handle:
            handle.write(best["foldseek_3di"])

    print("[OK] %s" % dataset_name)
    print("  csv:      %s" % csv_path)
    print("  pdb:      %s" % pdb_path)
    print("  chain:    %s" % best["name"])
    print("  identity: %.3f" % best["identity_to_wt"])
    print("  coverage: %.3f" % best["coverage_to_wt"])
    if save_output:
        print("  wrote:    %s" % output_path)
    if save_text_files:
        print("  wrote:    %s" % pdb_sequence_txt_path)
        print("  wrote:    %s" % foldseek_token_sequence_txt_path)

    if print_tokens:
        print("\n[pdb_sequence]")
        print(best["aa_seq"])
        print("\n[pdb_foldseek_3di]")
        print(best["foldseek_3di"])
        print("\n[aligned_3di_to_wt]")
        print(best["aligned_3di_to_wt"])
        print("\n[saprot_wt_sequence]")
        print(saprot_wt_sequence)

    return out


def _apply_simple_argv_defaults():
    dataset_name = DATASET_NAME
    csv_path = CSV_PATH
    pdb_path = PDB_PATH

    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        csv_path = sys.argv[2]
    if len(sys.argv) > 3:
        pdb_path = sys.argv[3]

    return dataset_name, csv_path, pdb_path


if __name__ == "__main__":
    dataset_name, csv_path, pdb_path = _apply_simple_argv_defaults()
    make_foldseek_tokens_for_dataset(
        dataset_name=dataset_name,
        csv_path=csv_path,
        pdb_path=pdb_path,
        output_path=OUTPUT_PATH,
        save_output=SAVE_OUTPUT,
        save_text_files=SAVE_TEXT_FILES,
        print_tokens=PRINT_TOKENS,
    )
