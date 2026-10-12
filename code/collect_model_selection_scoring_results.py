#!/usr/bin/env python3
"""Collect sequence scorer outputs into five aggregate CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLLECTIONS = {
    "by_mutation_val_less": ("by_mutation", "less_than_K", "evaluation_on_*.csv"),
    "by_mutation_val_eq": ("by_mutation", "equal_to_K", "evaluation_on_*.csv"),
    "by_mutation_val_full": ("by_mutation", "full", "evaluation_on_*.csv"),
    "random_full": ("random", "full", "evaluation_train_size_*.csv"),
    "random_internal": ("random", "random_internal", "evaluation_train_size_*.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_path",
        required=True,
        help=(
            "Path to the scorer label cache, usually "
            ".../scoring_cache/regressor or .../scoring_cache/classifier..."
        ),
    )
    parser.add_argument(
        "--output_dir",
        help="Directory for collected CSVs. Defaults to <cache_path>/results/collected.",
    )
    parser.add_argument("--allow_empty", action="store_true")
    return parser.parse_args()


def result_root(cache_path: Path) -> Path:
    if (cache_path / "results").is_dir():
        return cache_path / "results"
    raise FileNotFoundError(f"could not find results directory under {cache_path}")


def collect_one(results_root: Path, group: str, resolution: str, pattern: str) -> pd.DataFrame:
    root = results_root / group / resolution
    frames = []
    if not root.is_dir():
        return pd.DataFrame()

    for path in sorted(root.glob(f"*/{pattern}")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame.insert(0, "source_file", str(path))
        frame.insert(0, "source_feature", path.parent.name)
        frame.insert(0, "source_resolution", resolution)
        frame.insert(0, "source_group", group)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> int:
    args = parse_args()
    cache_path = Path(args.cache_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else cache_path / "results" / "collected"
    results_root = result_root(cache_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for output_name, (group, resolution, pattern) in COLLECTIONS.items():
        collected = collect_one(results_root, group, resolution, pattern)
        if collected.empty and not args.allow_empty:
            print(f"[collect] warning: no rows found for {output_name}")
        out_path = output_dir / f"{output_name}.csv"
        collected.to_csv(out_path, index=False)
        summary_rows.append(
            {
                "collection": output_name,
                "group": group,
                "resolution": resolution,
                "pattern": pattern,
                "rows": len(collected),
                "path": str(out_path),
            }
        )
        print(f"[collect] wrote {out_path} rows={len(collected)}")

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[collect] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
