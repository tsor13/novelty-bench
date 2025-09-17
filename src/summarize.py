import argparse
import json
import os

import pandas as pd


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _compute_stats(values: pd.Series) -> tuple[float | None, float | None, float | None]:
    if values.empty:
        return None, None, None

    count = len(values)
    mean = float(values.mean())

    if count == 1:
        std = 0.0
    else:
        std = float(values.std(ddof=1))

    sem = float(std / (count ** 0.5)) if count > 0 else None
    return mean, std, sem


def _add_summary(summary: dict, key: str, values: pd.Series) -> None:
    mean, std, sem = _compute_stats(values)
    summary[key] = mean
    summary[f"{key}_std"] = std
    summary[f"{key}_sem"] = sem


def summarize(df: pd.DataFrame) -> dict:
    summary: dict[str, float | None] = {}

    partition_lengths = df["partition_scores"].map(len)
    _add_summary(summary, "mean_distinct", _numeric_series(partition_lengths))

    utility_series = _numeric_series(df["utility"])
    _add_summary(summary, "mean_utility", utility_series)

    generation_values = pd.Series(dtype=float)
    if "mean_generation_reward" in df.columns:
        generation_values = _numeric_series(df["mean_generation_reward"])
    elif "generation_rewards" in df.columns:
        generation_values = _numeric_series(df["generation_rewards"].explode())
    _add_summary(summary, "mean_generation_reward", generation_values)

    raw_generation_values = pd.Series(dtype=float)
    if "mean_raw_generation_reward" in df.columns:
        raw_generation_values = _numeric_series(df["mean_raw_generation_reward"])
    elif "raw_generation_rewards" in df.columns:
        raw_generation_values = _numeric_series(df["raw_generation_rewards"].explode())
    _add_summary(summary, "mean_raw_generation_reward", raw_generation_values)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    scores_path = os.path.join(eval_dir, "scores.jsonl")
    df = pd.read_json(scores_path, lines=True)

    summary = summarize(df)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
