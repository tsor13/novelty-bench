import argparse
import json
import os

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame, generation_df: pd.DataFrame | None = None) -> dict:
    summary = {}
    summary["mean_distinct"] = float(np.mean(df["partition_scores"].map(len)))
    summary["mean_utility"] = float(np.mean(df["utility"]))

    if generation_df is not None and not generation_df.empty:
        generation_series = generation_df["generation_rewards"].explode()
        generation_series = pd.to_numeric(generation_series, errors="coerce").dropna()
        if not generation_series.empty:
            summary["mean_generation_reward"] = float(generation_series.mean())

        if "raw_generation_rewards" in generation_df.columns:
            raw_series = generation_df["raw_generation_rewards"].explode()
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

    generation_path = os.path.join(eval_dir, "generation_rewards.jsonl")
    generation_df = None
    if os.path.exists(generation_path):
        generation_df = pd.read_json(generation_path, lines=True)

    summary = summarize(df, generation_df)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
