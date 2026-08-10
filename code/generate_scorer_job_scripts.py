#!/usr/bin/env python3
"""Generate bash scripts for sequence_embedding_scoring_analysis.py runs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
from pathlib import Path


VALID_MUTATION_RESOLUTIONS = ("full", "less_than_K", "equal_to_K", "specific_K")
VALID_RANDOM_RESOLUTIONS = ("full", "random_internal", "specific_K")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sequence_path", "--dataset_path", dest="sequence_path", required=True)
    parser.add_argument("--cache_path")
    parser.add_argument(
        "--scorer_script",
        default=str(Path(__file__).resolve().with_name("sequence_embedding_scoring_analysis.py")),
    )
    parser.add_argument("--conda_env")
    parser.add_argument("--python_executable", default="python")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--min_nmuts", type=int, required=True)
    parser.add_argument("--max_nmuts", type=int, required=True)
    parser.add_argument("--min_train_muts", "--min_train_nmuts", dest="min_train_muts", type=int)
    parser.add_argument("--max_train_muts", "--max_train_nmuts", dest="max_train_muts", type=int)
    parser.add_argument("--max_test_muts", "--max_test_nmuts", dest="max_test_muts", type=int)
    parser.add_argument("--num_muts_colname", default="num_muts")
    parser.add_argument("--activity_column_name", "--activity_col_name", dest="activity_column_name", required=True)
    parser.add_argument("--first_col", "--first_mutation_col", dest="first_col")
    parser.add_argument("--last_col", "--last_mutation_col", dest="last_col")

    parser.add_argument("--embeddings", "--embedding", dest="embeddings", nargs="*", default=[])
    parser.add_argument("--no_onehot", action="store_true")
    parser.add_argument("--mean_embeddings", dest="mean_embeddings", action="store_true", default=True)
    parser.add_argument("--flat_embeddings", dest="mean_embeddings", action="store_false")
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--load_all", action="store_true")
    parser.add_argument("--maximum_embeddings_to_load", type=int, default=20000)
    parser.add_argument("--cache_embedding_chunks", action="store_true")

    label = parser.add_mutually_exclusive_group(required=True)
    label.add_argument("--regressor", action="store_true")
    label.add_argument("--classifier", action="store_true")
    label.add_argument("--classifier_percentile", type=float)
    label.add_argument("--classifier_value", type=float)

    parser.add_argument("--train_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--niters", type=int, default=10)
    parser.add_argument("--validation_niters", type=int, default=5)
    parser.add_argument("--validation_fraction_split", type=float, nargs="*", default=None)
    parser.add_argument("--validation_train_sizes", type=int, nargs="*", default=[])
    parser.add_argument("--random_internal_validation_fraction_split", type=float, nargs="*", default=[0.5, 0.75])
    parser.add_argument("--random_internal_min_val_points", type=int, default=5)
    parser.add_argument("--validation_niters_full", type=int, default=5)
    parser.add_argument(
        "--validation_fraction_split_full",
        type=float,
        nargs="*",
        default=None,
    )
    parser.add_argument("--validation_train_sizes_full", type=int, nargs="*", default=[])
    parser.add_argument("--architecture_max_epochs", type=int, default=300)
    parser.add_argument("--architecture_eval_every", type=int, default=20)
    parser.add_argument("--architecture_max_steps", type=int)
    parser.add_argument("--architecture_eval_every_steps", type=int)
    parser.add_argument("--architecture_fine_max_steps", type=int)
    parser.add_argument("--architecture_fine_eval_every_steps", type=int)
    parser.add_argument("--architecture_n_seeds", type=int, default=1)
    parser.add_argument("--two_stage_architecture_resolution", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--architecture_top_k", type=int, default=3)
    parser.add_argument("--architecture_fine_batch_size", type=int, default=128)
    parser.add_argument("--architecture_final_batch_size", type=int, default=64)
    parser.add_argument("--architecture_final_lr_scale", type=float, default=math.sqrt(2.0))
    parser.add_argument("--auto_architecture_batch_size", action="store_true")
    parser.add_argument("--precision_k", type=int, default=100)
    parser.add_argument("--indistinguishable_tolerance", type=float, default=0.01)
    parser.add_argument("--include_specific_k", action="store_true")
    parser.add_argument("--specific_k_values", type=int, nargs="*")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--refresh_architecture", action="store_true")
    parser.add_argument("--hard_refresh", "--hard-refresh", dest="hard_refresh", action="store_true")
    parser.add_argument(
        "--scoring_only",
        "--scoring-only",
        action="store_true",
        help="Only write score_after_resolving scripts and scoring_manifest.csv; do not write preflight or architecture scripts.",
    )
    parser.add_argument("--verbose_debug_prints", action="store_true")
    return parser.parse_args()


def quote_command(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def command_prefix(args):
    if args.conda_env:
        return ["conda", "run", "--no-capture-output", "-n", args.conda_env, args.python_executable]
    return [args.python_executable]


def base_args(args, include_refresh_architecture=True):
    parts = [
        "--dataset_path",
        str(Path(args.sequence_path).expanduser()),
        "--activity_column_name",
        args.activity_column_name,
        "--num_muts_colname",
        args.num_muts_colname,
        "--niters",
        args.niters,
        "--validation_niters",
        args.validation_niters,
        "--random_internal_validation_fraction_split",
        *args.random_internal_validation_fraction_split,
        "--random_internal_min_val_points",
        args.random_internal_min_val_points,
        "--validation_niters_full",
        args.validation_niters_full,
        "--architecture_max_epochs",
        args.architecture_max_epochs,
        "--architecture_eval_every",
        args.architecture_eval_every,
        "--architecture_n_seeds",
        args.architecture_n_seeds,
        "--architecture_top_k",
        args.architecture_top_k,
        "--architecture_fine_batch_size",
        args.architecture_fine_batch_size,
        "--architecture_final_batch_size",
        args.architecture_final_batch_size,
        "--architecture_final_lr_scale",
        args.architecture_final_lr_scale,
        "--precision_k",
        args.precision_k,
        "--indistinguishable_tolerance",
        args.indistinguishable_tolerance,
        "--maximum_embeddings_to_load",
        args.maximum_embeddings_to_load,
        "--random_seed",
        args.random_seed,
        "--device",
        args.device,
    ]
    if args.architecture_max_steps is not None:
        parts.extend(["--architecture_max_steps", args.architecture_max_steps])
    if args.architecture_eval_every_steps is not None:
        parts.extend(["--architecture_eval_every_steps", args.architecture_eval_every_steps])
    if args.architecture_fine_max_steps is not None:
        parts.extend(["--architecture_fine_max_steps", args.architecture_fine_max_steps])
    if args.architecture_fine_eval_every_steps is not None:
        parts.extend(["--architecture_fine_eval_every_steps", args.architecture_fine_eval_every_steps])
    if not args.two_stage_architecture_resolution:
        parts.append("--no-two_stage_architecture_resolution")
    if args.validation_fraction_split is not None:
        parts.extend(["--validation_fraction_split", *args.validation_fraction_split])
    if args.validation_train_sizes:
        parts.extend(["--validation_train_sizes", *args.validation_train_sizes])
    if args.validation_fraction_split_full is not None:
        parts.extend(["--validation_fraction_split_full", *args.validation_fraction_split_full])
    if args.validation_train_sizes_full:
        parts.extend(["--validation_train_sizes_full", *args.validation_train_sizes_full])
    if args.auto_architecture_batch_size:
        parts.append("--auto_architecture_batch_size")
    if args.cache_path:
        parts.extend(["--cache_path", str(Path(args.cache_path).expanduser())])
    if args.regressor:
        parts.append("--regressor")
    elif args.classifier:
        parts.append("--classifier")
    elif args.classifier_percentile is not None:
        parts.extend(["--classifier_percentile", args.classifier_percentile])
    elif args.classifier_value is not None:
        parts.extend(["--classifier_value", args.classifier_value])
    if include_refresh_architecture and args.refresh_architecture:
        parts.append("--refresh_architecture")
    if args.verbose_debug_prints:
        parts.append("--verbose_debug_prints")
    return parts


def feature_specs(args):
    specs = []
    if not args.no_onehot:
        if not args.first_col or not args.last_col:
            raise ValueError("one-hot generation requires --first_col and --last_col; pass --no_onehot to skip it")
        specs.append(
            {
                "name": "onehot",
                "args": [
                    "--onehot",
                    "--first_mutation_col",
                    args.first_col,
                    "--last_mutation_col",
                    args.last_col,
                ],
            }
        )
    for embedding in args.embeddings:
        emb_args = ["--embedding", embedding]
        if args.mean_embeddings:
            emb_args.append("--mean_embeddings")
        if args.normalize_embeddings:
            emb_args.append("--normalize_embeddings")
        if args.load_all:
            emb_args.append("--load_all")
        if args.cache_embedding_chunks:
            emb_args.append("--cache_embedding_chunks")
        embedding_prefix = "mean" if args.mean_embeddings else "flat"
        specs.append({"name": sanitize(f"{embedding_prefix}_{embedding}"), "args": emb_args})
    if not specs:
        raise ValueError("nothing to generate: enable one-hot or pass at least one --embeddings value")
    return specs


def min_train_muts(args):
    return int(args.min_train_muts if args.min_train_muts is not None else args.min_nmuts)


def max_test_muts(args):
    return int(args.max_test_muts if args.max_test_muts is not None else args.max_nmuts)


def max_train_muts(args):
    if args.max_train_muts is not None:
        return int(args.max_train_muts)
    return max_test_muts(args) - 1


def architecture_specs(args):
    specs = []
    specs.append(
        {
            "train_type": "",
            "resolution": "full",
            "specific_k": "",
            "resolve_k": "",
            "train_size": "",
            "split_iteration": "",
            "name": "full",
            "args": ["--architecture_resolution", "full", "--resolve_architecture_only"],
        }
    )

    for resolution in ("less_than_K", "equal_to_K"):
        for k in range(min_train_muts(args), max_train_muts(args) + 1):
            specs.append(
                {
                    "train_type": "",
                    "resolution": resolution,
                    "specific_k": "",
                    "resolve_k": k,
                    "train_size": "",
                    "split_iteration": "",
                    "name": f"{resolution}_K_{k}",
                    "args": mutation_bound_args(args) + [
                        "--architecture_resolution",
                        resolution,
                        "--resolve_architecture_only",
                        "--resolve_k",
                        k,
                    ],
                }
            )

    if args.include_specific_k:
        for specific_k in specific_k_values(args):
            specs.append(
                {
                    "train_type": "",
                    "resolution": "specific_K",
                    "specific_k": specific_k,
                    "resolve_k": specific_k,
                    "train_size": "",
                    "split_iteration": "",
                    "name": f"specific_K_{specific_k}",
                    "args": mutation_bound_args(args) + [
                        "--architecture_resolution",
                        "specific_K",
                        "--resolve_architecture_only",
                        "--resolve_k",
                        specific_k,
                    ],
                }
            )

    if args.train_sizes:
        for size in args.train_sizes:
            for iteration in range(1, int(args.niters) + 1):
                specs.append(
                    {
                        "train_type": "",
                        "resolution": "random_internal",
                        "specific_k": "",
                        "resolve_k": "",
                        "train_size": int(size),
                        "split_iteration": int(iteration),
                        "name": f"random_internal_size_{int(size)}_iter_{int(iteration):03d}",
                        "args": [
                            "--architecture_resolution",
                            "random_internal",
                            "--resolve_architecture_only",
                            "--resolve_train_size",
                            int(size),
                            "--resolve_split_iteration",
                            int(iteration),
                        ],
                    }
                )
    return specs


def scoring_specs(args):
    specs = []
    for resolution in VALID_MUTATION_RESOLUTIONS:
        if resolution == "specific_K":
            if not args.include_specific_k:
                continue
            for specific_k in specific_k_values(args):
                for k in range(min_train_muts(args), max_train_muts(args) + 1):
                    specs.append(
                        {
                            "train_type": "mutation",
                            "resolution": resolution,
                            "specific_k": specific_k,
                            "resolve_k": "",
                            "score_k": k,
                            "train_size": "",
                            "split_iteration": "",
                            "name": f"mutation_specific_K_{specific_k}_train_K_{k}",
                            "args": mutation_args(args, resolution)
                            + ["--specific_k", specific_k, "--score_k", k],
                        }
                    )
        else:
            for k in range(min_train_muts(args), max_train_muts(args) + 1):
                specs.append(
                    {
                        "train_type": "mutation",
                        "resolution": resolution,
                        "specific_k": "",
                        "resolve_k": "",
                        "score_k": k,
                        "train_size": "",
                        "split_iteration": "",
                        "name": f"mutation_{resolution}_train_K_{k}",
                        "args": mutation_args(args, resolution) + ["--score_k", k],
                    }
                )

    if args.train_sizes:
        for resolution in VALID_RANDOM_RESOLUTIONS:
            if resolution == "specific_K":
                if not args.include_specific_k:
                    continue
                for specific_k in specific_k_values(args):
                    for size in args.train_sizes:
                        for iteration in range(1, int(args.niters) + 1):
                            specs.append(
                                {
                                    "train_type": "random",
                                    "resolution": resolution,
                                    "specific_k": specific_k,
                                    "resolve_k": "",
                                    "score_k": "",
                                    "train_size": int(size),
                                    "split_iteration": int(iteration),
                                    "name": (
                                        f"random_specific_K_{specific_k}_"
                                        f"size_{int(size)}_iter_{int(iteration):03d}"
                                    ),
                                    "args": random_args(args, resolution, [int(size)])
                                    + [
                                        "--specific_k",
                                        specific_k,
                                        "--score_train_size",
                                        int(size),
                                        "--score_split_iteration",
                                        int(iteration),
                                    ],
                                }
                            )
            else:
                for size in args.train_sizes:
                    for iteration in range(1, int(args.niters) + 1):
                        specs.append(
                            {
                                "train_type": "random",
                                "resolution": resolution,
                                "specific_k": "",
                                "resolve_k": "",
                                "score_k": "",
                                "train_size": int(size),
                                "split_iteration": int(iteration),
                                "name": f"random_{resolution}_size_{int(size)}_iter_{int(iteration):03d}",
                                "args": random_args(args, resolution, [int(size)])
                                + [
                                    "--score_train_size",
                                    int(size),
                                    "--score_split_iteration",
                                    int(iteration),
                                ],
                            }
                        )
    return specs


def mutation_bound_args(args):
    parts = [
        "--min_muts",
        args.min_nmuts,
        "--max_muts",
        args.max_nmuts,
    ]
    if args.min_train_muts is not None:
        parts.extend(["--min_train_muts", args.min_train_muts])
    if args.max_train_muts is not None:
        parts.extend(["--max_train_muts", args.max_train_muts])
    if args.max_test_muts is not None:
        parts.extend(["--max_test_muts", args.max_test_muts])
    return parts


def mutation_args(args, resolution):
    parts = [
        "--train_type",
        "mutation",
        *mutation_bound_args(args),
        "--architecture_resolution",
        resolution,
    ]
    return parts


def random_args(args, resolution, train_sizes=None):
    if train_sizes is None:
        train_sizes = args.train_sizes
    return [
        "--train_type",
        "random",
        "--train_sizes",
        *train_sizes,
        "--architecture_resolution",
        resolution,
    ]


def specific_k_values(args):
    if args.specific_k_values:
        return [int(value) for value in args.specific_k_values]
    return list(range(min_train_muts(args), max_test_muts(args) + 1))


def write_script(path, command, *, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"echo {shlex.quote(title)}",
                quote_command(command),
                "",
            ]
        )
    )
    os.chmod(path, 0o755)


def write_manifest(path, rows):
    fieldnames = [
        "script",
        "feature",
        "train_type",
        "architecture_resolution",
        "specific_k",
        "resolve_k",
        "score_k",
        "min_train_muts",
        "max_train_muts",
        "max_test_muts",
        "train_size",
        "split_iteration",
        "command",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize(value):
    text = str(value).strip()
    safe = []
    for char in text:
        safe.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(safe).strip("._") or "value"


def count_csv_data_rows(path):
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def filter_random_train_sizes(args):
    n_rows = count_csv_data_rows(args.sequence_path)
    if n_rows is None:
        print(f"[generator] sequence CSV not found locally; random train sizes were not prevalidated: {args.sequence_path}")
        return
    valid = []
    skipped = []
    for size in args.train_sizes:
        size = int(size)
        if 1 <= size < n_rows:
            valid.append(size)
        else:
            skipped.append(size)
    if skipped:
        print(
            "[generator] skipped invalid random train sizes "
            f"for n_rows={n_rows}: {skipped}"
        )
    args.train_sizes = valid


def make_preflight_script(args, output_dir, features):
    commands = []
    prefix = command_prefix(args)
    scorer = str(Path(args.scorer_script).expanduser())
    common = base_args(args)
    split_write_flags = ["--make_splits_only"]
    if args.refresh:
        split_write_flags.append("--refresh")
    hard_refresh_pending = True
    if args.train_sizes:
        random_flags = list(split_write_flags)
        if hard_refresh_pending:
            random_flags.append("--hard-refresh")
            hard_refresh_pending = False
        random_split_args = [
            "--train_type",
            "random",
            "--train_sizes",
            *args.train_sizes,
            "--architecture_resolution",
            "full",
            *random_flags,
        ]
        commands.append(prefix + [scorer, *common, *features[0]["args"], *random_split_args])
    mutation_flags = list(split_write_flags)
    if hard_refresh_pending:
        mutation_flags.append("--hard-refresh")
    mutation_split_args = [
        "--train_type",
        "mutation",
        "--min_muts",
        args.min_nmuts,
        "--max_muts",
        args.max_nmuts,
        "--architecture_resolution",
        "full",
        *mutation_flags,
    ]
    if args.min_train_muts is not None:
        mutation_split_args.extend(["--min_train_muts", args.min_train_muts])
    if args.max_train_muts is not None:
        mutation_split_args.extend(["--max_train_muts", args.max_train_muts])
    if args.max_test_muts is not None:
        mutation_split_args.extend(["--max_test_muts", args.max_test_muts])
    commands.append(prefix + [scorer, *common, *features[0]["args"], *mutation_split_args])

    path = output_dir / "preflight" / "00_make_splits.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "echo making scorer splits"]
    lines.extend(quote_command(command) for command in commands)
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def main():
    args = parse_args()
    if args.load_all:
        args.maximum_embeddings_to_load = -1
        print("[generator] --load_all enabled; generated scripts will load embeddings all at once")
    requested_train_sizes = list(args.train_sizes)
    filter_random_train_sizes(args)
    if args.max_nmuts < args.min_nmuts:
        raise ValueError("--max_nmuts must be >= --min_nmuts")
    if max_train_muts(args) < min_train_muts(args):
        raise ValueError("--max_train_muts must be >= --min_train_muts")
    if max_test_muts(args) <= max_train_muts(args):
        raise ValueError("--max_test_muts must be greater than --max_train_muts")
    if args.precision_k <= 0:
        raise ValueError("--precision_k must be positive")
    if args.architecture_top_k <= 0:
        raise ValueError("--architecture_top_k must be positive")
    if args.architecture_fine_batch_size <= 0:
        raise ValueError("--architecture_fine_batch_size must be positive")
    if args.architecture_final_batch_size <= 0:
        raise ValueError("--architecture_final_batch_size must be positive")
    if args.architecture_final_lr_scale <= 0:
        raise ValueError("--architecture_final_lr_scale must be positive")
    if args.architecture_max_steps is not None and args.architecture_max_steps <= 0:
        raise ValueError("--architecture_max_steps must be positive")
    if args.architecture_eval_every_steps is not None and args.architecture_eval_every_steps <= 0:
        raise ValueError("--architecture_eval_every_steps must be positive")
    if args.architecture_fine_max_steps is not None and args.architecture_fine_max_steps <= 0:
        raise ValueError("--architecture_fine_max_steps must be positive")
    if args.architecture_fine_eval_every_steps is not None and args.architecture_fine_eval_every_steps <= 0:
        raise ValueError("--architecture_fine_eval_every_steps must be positive")
    if args.hard_refresh and args.scoring_only:
        raise ValueError("--hard-refresh is incompatible with --scoring_only because scoring-only does not write splits")
    if args.hard_refresh and not args.refresh:
        print("[generator] --hard-refresh will delete the active scorer cache before preflight split creation")
    if not Path(args.scorer_script).expanduser().is_file():
        raise FileNotFoundError(f"scorer script not found: {args.scorer_script}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    architecture_dir = output_dir / "resolve_architectures"
    scoring_dir = output_dir / "score_after_resolving"
    features = feature_specs(args)
    architecture_jobs = architecture_specs(args)
    scoring_jobs = scoring_specs(args)
    if not args.train_sizes and requested_train_sizes:
        print("[generator] no valid --train_sizes remain; random train_type scripts were skipped")
    elif not args.train_sizes:
        print("[generator] --train_sizes not provided; random train_type scripts were skipped")

    prefix = command_prefix(args)
    scorer = str(Path(args.scorer_script).expanduser())
    architecture_common = base_args(args, include_refresh_architecture=True)
    scoring_common = base_args(args, include_refresh_architecture=False) + ["--require_resolved_architecture"]
    architecture_rows = []
    scoring_rows = []
    if args.scoring_only:
        print("[generator] --scoring_only enabled; skipping preflight and architecture scripts")
    else:
        counter = 1
        for feature in features:
            for job in architecture_jobs:
                name = f"{counter:04d}_{feature['name']}_{job['name']}.sh"
                script_path = architecture_dir / name
                command = prefix + [scorer, *architecture_common, *feature["args"], *job["args"]]
                write_script(
                    script_path,
                    command,
                    title=f"resolving architecture {feature['name']} {job['resolution']}",
                )
                architecture_rows.append(
                    {
                        "script": str(script_path),
                        "feature": feature["name"],
                        "train_type": job["train_type"],
                        "architecture_resolution": job["resolution"],
                        "specific_k": job["specific_k"],
                        "resolve_k": job["resolve_k"],
                        "score_k": "",
                        "min_train_muts": min_train_muts(args),
                        "max_train_muts": max_train_muts(args),
                        "max_test_muts": max_test_muts(args),
                        "train_size": job["train_size"],
                        "split_iteration": job["split_iteration"],
                        "command": quote_command(command),
                    }
                )
                counter += 1

    counter = 1
    for feature in features:
        for job in scoring_jobs:
            name = f"{counter:04d}_{feature['name']}_{job['name']}.sh"
            script_path = scoring_dir / name
            command = prefix + [scorer, *scoring_common, *feature["args"], *job["args"]]
            write_script(
                script_path,
                command,
                title=f"scoring {feature['name']} {job['train_type']} {job['resolution']}",
            )
            scoring_rows.append(
                {
                    "script": str(script_path),
                    "feature": feature["name"],
                    "train_type": job["train_type"],
                    "architecture_resolution": job["resolution"],
                    "specific_k": job["specific_k"],
                    "resolve_k": job["resolve_k"],
                    "score_k": job["score_k"],
                    "min_train_muts": min_train_muts(args),
                    "max_train_muts": max_train_muts(args),
                    "max_test_muts": max_test_muts(args),
                    "train_size": job["train_size"],
                    "split_iteration": job["split_iteration"],
                    "command": quote_command(command),
                }
            )
            counter += 1

    architecture_manifest = output_dir / "architecture_manifest.csv"
    scoring_manifest = output_dir / "scoring_manifest.csv"
    if not args.scoring_only:
        preflight = make_preflight_script(args, output_dir, features)
        write_manifest(architecture_manifest, architecture_rows)
    write_manifest(scoring_manifest, scoring_rows)

    if not args.scoring_only:
        print(f"[generator] wrote {len(architecture_rows)} architecture scripts to {architecture_dir}")
    print(f"[generator] wrote {len(scoring_rows)} scoring scripts to {scoring_dir}")
    if not args.scoring_only:
        print(f"[generator] wrote split preflight script to {preflight}")
        print(f"[generator] wrote architecture manifest to {architecture_manifest}")
    print(f"[generator] wrote scoring manifest to {scoring_manifest}")


if __name__ == "__main__":
    main()
