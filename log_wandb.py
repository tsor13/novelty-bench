#!/usr/bin/env python3
"""
Log summarized novelty-bench results to Weights & Biases projects.

export WANDB_ENTITY=tsor1313

uv run log_wandb.py --dry-run
uv run log_wandb.py

"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import wandb
except ImportError as exc:  # pragma: no cover - surfaced at runtime
    raise SystemExit(
        "wandb is required to run this script. Install it with `pip install wandb`."
    ) from exc


@dataclass(frozen=True)
class SummaryItem:
    summary_path: Path
    model_name: str
    split: str
    template_mode: str
    sampling: str

    def run_name(self) -> str:
        base = f"{self.model_name}__{self.template_mode}__{self.sampling}"
        return slugify(base)

    def project_name(self, prefix: str) -> str:
        return f"{prefix}-{self.split}"


def slugify(value: str) -> str:
    """Convert arbitrary text into a run-name-friendly slug."""
    value = value.strip().replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value)


def parse_summary_files(results_dir: Path) -> Iterator[SummaryItem]:
    """Yield metadata for each summary.json under the results directory."""
    for summary_path in sorted(results_dir.rglob("summary.json")):
        rel_parts = summary_path.relative_to(results_dir).parts
        if len(rel_parts) < 4:
            print(f"WARN: Skipping {summary_path}: expected at least 4 path components", file=sys.stderr)
            continue

        split, template_mode, sampling = rel_parts[-4:-1]
        model_parts = rel_parts[:-4]
        if not model_parts:
            print(f"WARN: Skipping {summary_path}: could not infer model name", file=sys.stderr)
            continue

        model_name = "/".join(model_parts)
        yield SummaryItem(
            summary_path=summary_path,
            model_name=model_name,
            split=split,
            template_mode=template_mode,
            sampling=sampling,
        )


def load_metrics(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    metrics: Dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
            continue
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            print(f"WARN: Skipping non-numeric metric {key} in {path}", file=sys.stderr)

    return metrics


def fetch_existing_keys(api: wandb.Api, entity: Optional[str], project: str) -> Set[Tuple[str, str, str]]:
    """Collect already-logged (model, template, sampling) triples for a project."""
    project_path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(project_path, per_page=200)
    except wandb.errors.CommError as exc:
        print(f"WARN: Could not list runs for {project_path}: {exc}", file=sys.stderr)
        return set()

    try:
        iterable = list(runs)
    except ValueError as exc:
        print(f"INFO: {exc}. Treating as empty project.", file=sys.stderr)
        return set()

    existing: Set[Tuple[str, str, str]] = set()
    for run in iterable:
        config = run.config or {}
        key = (
            config.get("model_name"),
            config.get("template_mode"),
            config.get("sampling"),
        )
        if all(key):
            existing.add(key)  # type: ignore[arg-type]
    return existing


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("new_results"),
        help="Directory containing model evaluation outputs (default: new_results)",
    )
    parser.add_argument(
        "--entity",
        help="W&B entity (team or user). Defaults to the configured W&B account.",
    )
    parser.add_argument(
        "--project-prefix",
        default="noveltybench",
        help="Base project prefix. Final project is <prefix>-<split> (default: noveltybench)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-log runs even if an entry with the same model/template/sampling already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without sending anything to W&B.",
    )
    args = parser.parse_args(argv)

    if not args.results_dir.is_dir():
        raise SystemExit(f"Results directory {args.results_dir} not found")

    summaries = list(parse_summary_files(args.results_dir))
    if not summaries:
        print("No summary.json files found. Nothing to log.")
        return 0

    api = wandb.Api() if not args.force else None
    existing_cache: Dict[str, Set[Tuple[str, str, str]]] = {}

    logged = 0
    skipped = 0

    for item in summaries:
        project = item.project_name(args.project_prefix)
        key = (item.model_name, item.template_mode, item.sampling)

        if not args.force:
            if project not in existing_cache:
                if api is None:
                    api = wandb.Api()
                existing_cache[project] = fetch_existing_keys(api, args.entity, project)
            if key in existing_cache.get(project, set()):
                print(f"SKIP: Skipping {item.summary_path}: already logged in {project}")
                skipped += 1
                continue

        metrics = load_metrics(item.summary_path)
        config = {
            "model_name": item.model_name,
            "split": item.split,
            "template_mode": item.template_mode,
            "sampling": item.sampling,
        }

        if args.dry_run:
            print(
                f"DRY RUN Would log {item.summary_path} to project {project} "
                f"as run {item.run_name()} with metrics {metrics}"
            )
            logged += 1
            continue

        run = wandb.init(
            project=project,
            entity=args.entity,
            name=item.run_name(),
            config=config,
            reinit=True,
        )
        run.summary.update(metrics)
        run.finish()
        logged += 1

        if not args.force:
            existing_cache.setdefault(project, set()).add(key)

    print(f"Done. Logged {logged} runs, skipped {skipped} duplicates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
