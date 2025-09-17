import argparse
import json
import os

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> dict:
    summary = {}
    summary["mean_distinct"] = float(np.mean(df["partition_scores"].map(len)))
    summary["mean_utility"] = float(np.mean(df["utility"]))

    summary["mean_generation_reward"] = None
    if "mean_generation_reward" in df.columns:
        gen_mean_series = pd.to_numeric(df["mean_generation_reward"], errors="coerce").dropna()
        if not gen_mean_series.empty:
            summary["mean_generation_reward"] = float(gen_mean_series.mean())
    elif "generation_rewards" in df.columns:
        generation_series = df["generation_rewards"].explode()
        generation_series = pd.to_numeric(generation_series, errors="coerce").dropna()
        if not generation_series.empty:
            summary["mean_generation_reward"] = float(generation_series.mean())

    summary["mean_raw_generation_reward"] = None
    if "mean_raw_generation_reward" in df.columns:
        raw_mean_series = pd.to_numeric(df["mean_raw_generation_reward"], errors="coerce").dropna()
        if not raw_mean_series.empty:
            summary["mean_raw_generation_reward"] = float(raw_mean_series.mean())
    elif "raw_generation_rewards" in df.columns:
        raw_series = df["raw_generation_rewards"].explode()
        raw_series = pd.to_numeric(raw_series, errors="coerce").dropna()
        if not raw_series.empty:
            summary["mean_raw_generation_reward"] = float(raw_series.mean())

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
