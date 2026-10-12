#!/usr/bin/env python3
"""Generate bash scripts for FuncScape feature-combo scoring splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path


DEFAULT_ARCH_FRACTIONS = (0.15, 0.25, 0.5, 0.75, 0.8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--funcscape_root", default=os.environ.get("FUNCSCAPE_ROOT", "."))
    p.add_argument("--features_pt", "--data_path", dest="data_path")
    p.add_argument("--output_dir")
    p.add_argument("--combo", action="append", nargs="+", default=[])
    p.add_argument("--y_key", default="activity")
    p.add_argument("--results_file_name", default="scoring_results.csv")
    p.add_argument("--scoring_experiment_name", default="scoring_experiment")
    p.add_argument("--subset_sizes", type=int, nargs="+", default=[500, 1000])
    p.add_argument("--n_iterations", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--conda_env")
    p.add_argument("--python_executable", default="python")
    p.add_argument("--architecture_resample_fractions", type=float, nargs="+", default=list(DEFAULT_ARCH_FRACTIONS))
    p.add_argument("--architecture_n_resamples_per_fraction", type=int, default=10)
    p.add_argument("--architecture_n_seeds", type=int, default=1)
    p.add_argument("--architecture_max_epochs", type=int, default=500)
    p.add_argument("--architecture_eval_every", type=int, default=10)
    p.add_argument("--precision_at_k", type=int)
    p.add_argument("--refresh_splits", action="store_true")
    p.add_argument("--refresh_cache", action="store_true")
    p.add_argument("--quiet", action="store_true")

    p.add_argument("--run_job_config")
    p.add_argument("--collect_manifest")
    p.add_argument("--output_csv")
    return p.parse_args()


def add_funcscape_to_path(funcscape_root):
    root = Path(funcscape_root).expanduser().resolve()
    for path in (root / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return root


def quote(parts):
    return " ".join(shlex.quote(str(x)) for x in parts)


def command_prefix(args):
    if args.conda_env:
        return ["conda", "run", "--no-capture-output", "-n", args.conda_env, args.python_executable]
    return [args.python_executable]


def normalize_combos(raw_combos):
    combos = []
    for raw in raw_combos:
        combo = []
        for item in raw:
            combo.extend(part for part in str(item).split(",") if part)
        combos.append(combo)
    if not combos:
        raise ValueError("Pass at least one --combo, e.g. --combo onehot or --combo onehot,esm")
    return combos


def sanitize(text):
    safe = [c if c.isalnum() or c in {"-", "_", "."} else "_" for c in str(text)]
    return "".join(safe).strip("._") or "value"


def write_script(path, command, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"echo {shlex.quote(title)}",
                quote(command),
                "",
            ]
        )
    )
    os.chmod(path, 0o755)


def write_manifest(path, rows):
    fieldnames = [
        "script",
        "feature_name",
        "feature_combo",
        "split_id",
        "cache_path",
        "config_path",
        "result_path",
        "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate(args):
    root = add_funcscape_to_path(args.funcscape_root)
    import src.experiments.run_scoring_experiment as rse

    data_path = Path(args.data_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    scripts_dir = output_dir / "jobs"
    configs_dir = output_dir / "job_configs"
    combos = normalize_combos(args.combo)

    features = rse._load_features(data_path, args.y_key)
    y, _task_type, target_scheme = rse._prepare_target(features[args.y_key])
    feature_names = [name for name in features if name != args.y_key]
    feature_ndims = rse._feature_ndims(features, feature_names)
    feature_combos = rse._build_feature_combinations(
        feature_names,
        all_singles=False,
        all_doubles=False,
        custom_combinations=combos,
        combine_all=False,
        feature_ndims=feature_ndims,
    )

    split_payload = rse.resolve_or_create_scoring_splits(
        data_path=data_path,
        target_scheme=target_scheme,
        n_rows=len(y),
        n_iterations=args.n_iterations,
        subset_sizes=args.subset_sizes,
        architecture_resample_fractions=args.architecture_resample_fractions,
        architecture_n_resamples_per_fraction=args.architecture_n_resamples_per_fraction,
        random_seed=args.random_seed,
        save_splits=True,
        refresh_splits=args.refresh_splits,
        scoring_experiment_name=args.scoring_experiment_name,
    )
    split_file_path = split_payload["path"]
    result_path = rse._resolve_results_path(
        args.results_file_name,
        data_path,
        args.scoring_experiment_name,
    )
    architecture_grid = [
        rse._json_safe_config(rse._architecture_config(config, idx))
        for idx, config in enumerate(rse.DEFAULT_ARCHITECTURE_GRID)
    ]

    rows = []
    runner = Path(__file__).resolve()
    counter = 1
    for combo_idx, combo_info in enumerate(feature_combos):
        for split_idx, split in enumerate(split_payload["splits"]):
            cache_path = rse._architecture_search_cache_path(
                data_path,
                target_scheme,
                combo_info,
                split["split_id"],
                args.scoring_experiment_name,
            )
            config = {
                "data_path": str(data_path),
                "y_key": args.y_key,
                "target_kwargs": {},
                "combo_info": combo_info,
                "split_file_path": str(split_file_path),
                "split_id": split["split_id"],
                "architecture_resamples": rse._normalize_architecture_resamples(split["architecture_resamples"]),
                "cache_path": str(cache_path),
                "result_path": str(result_path),
                "architecture_grid": architecture_grid,
                "max_epochs": int(args.architecture_max_epochs),
                "eval_every": int(args.architecture_eval_every),
                "resample_fractions": [float(x) for x in args.architecture_resample_fractions],
                "n_resamples_per_fraction": int(args.architecture_n_resamples_per_fraction),
                "n_seeds": int(args.architecture_n_seeds),
                "random_seed": int(args.random_seed + combo_idx * 1000 + split_idx),
                "precision_at_k": args.precision_at_k,
                "device": args.device,
                "verbose": not args.quiet,
                "refresh_cache": bool(args.refresh_cache),
            }
            name = f"{counter:04d}_{sanitize(combo_info['name'])}_{split['split_id']}"
            config_path = configs_dir / f"{name}.json"
            script_path = scripts_dir / f"{name}.bash"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            command = command_prefix(args) + [
                str(runner),
                "--funcscape_root",
                str(root),
                "--run_job_config",
                str(config_path),
            ]
            write_script(script_path, command, f"scoring {combo_info['name']} {split['split_id']}")
            rows.append(
                {
                    "script": str(script_path),
                    "feature_name": combo_info["name"],
                    "feature_combo": json.dumps(combo_info["combo"]),
                    "split_id": split["split_id"],
                    "cache_path": str(cache_path),
                    "config_path": str(config_path),
                    "result_path": str(result_path),
                    "command": quote(command),
                }
            )
            counter += 1

    manifest = output_dir / "scoring_manifest.csv"
    write_manifest(manifest, rows)
    collect_command = command_prefix(args) + [
        str(runner),
        "--collect_manifest",
        str(manifest),
        "--output_csv",
        str(result_path),
    ]
    write_script(output_dir / "collect_results.bash", collect_command, "collecting scoring results")
    print(f"[generator] wrote {len(rows)} scripts to {scripts_dir}")
    print(f"[generator] wrote manifest to {manifest}")
    print(f"[generator] wrote collector to {output_dir / 'collect_results.bash'}")
    print(f"[generator] split cache: {split_file_path}")
    print(f"[generator] result csv: {result_path}")


def run_job_config(args):
    root = add_funcscape_to_path(args.funcscape_root)
    import src.experiments.run_scoring_experiment as rse

    config = json.loads(Path(args.run_job_config).expanduser().read_text())
    cache_path = Path(config["cache_path"])
    if (
        cache_path.exists()
        and rse._architecture_search_cache_complete(cache_path, require_split_result=True)
        and not config.get("refresh_cache", False)
    ):
        print(f"[runner] already complete: {cache_path}")
        return
    os.chdir(root)
    rse.run_architecture_resolution_job(config)


def collect_results(args):
    import pandas as pd

    manifest = Path(args.collect_manifest).expanduser()
    rows = []
    with manifest.open(newline="") as handle:
        for item in csv.DictReader(handle):
            path = Path(item["cache_path"]) / "split_result.json"
            if not path.is_file():
                print(f"[collector] missing {path}")
                continue
            row = json.loads(path.read_text())
            row["cache_path"] = item["cache_path"]
            rows.append(row)
    output_csv = Path(args.output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[collector] wrote {len(rows)} rows to {output_csv}")


def main():
    args = parse_args()
    if args.run_job_config:
        run_job_config(args)
    elif args.collect_manifest:
        collect_results(args)
    else:
        if not args.data_path or not args.output_dir:
            raise ValueError("generation mode requires --features_pt and --output_dir")
        generate(args)


if __name__ == "__main__":
    main()
