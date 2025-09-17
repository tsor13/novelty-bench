import argparse
import asyncio
import json
import os
from typing import Tuple

import datasets
import numpy as np
import torch
from aiofiles import open as aio_open
from datasets import load_dataset
from tqdm.asyncio import tqdm

from score import (
    CONCURRENT_REQUESTS,
    rm_and_tokenizer,
    transform_raw_reward,
)


async def score_partition_all(
    prompt: str, generations: list[str], partition: list[int]
) -> Tuple[list[int], list[int], list[int], list[float]]:
    """Compute per-generation and partition rewards for a prompt."""
    rm, tokenizer = rm_and_tokenizer()
    convs = [
        [
            {"content": prompt, "role": "user"},
            {"content": generation, "role": "assistant"},
        ]
        for generation in generations
    ]
    batch = tokenizer.apply_chat_template(
        convs,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(rm.device)

    with torch.no_grad():
        raw_rewards = rm(**batch).logits[:, 0].tolist()

    generation_rewards = [transform_raw_reward(r) for r in raw_rewards]

    generation_scores = []
    partition_scores = []

    for score_value, partition_index in zip(
        generation_rewards, partition, strict=False
    ):
        if partition_index == len(partition_scores):
            generation_scores.append(score_value)
            partition_scores.append(score_value)
        else:
            generation_scores.append(0)

    assert len(partition_scores) == (max(partition) + 1), (
        f"partition_scores: {partition_scores}, partition: {partition}"
    )
    return generation_scores, partition_scores, generation_rewards, raw_rewards


async def process_instances(
    instances,
    scores_file,
    generation_rewards_file,
    patience,
):
    """Score all prompts and persist per-generation rewards."""
    if os.path.exists(scores_file) and os.path.exists(generation_rewards_file):
        try:
            scored = load_dataset("json", data_files=scores_file, split="train")
            generation_scored = load_dataset(
                "json", data_files=generation_rewards_file, split="train"
            )
            if not (set(instances["id"]) - set(scored["id"])) and not (
                set(instances["id"]) - set(generation_scored["id"])
            ):
                print("All prompts are scored. Skipping.")
                return
        except datasets.exceptions.DatasetGenerationError:
            pass

    async with aio_open(scores_file, "w", buffering=1) as f_scores:
        async with aio_open(generation_rewards_file, "w", buffering=1) as f_generations:
            semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

            async def process_single_instance(instance):
                async with semaphore:
                    (
                        generation_scores,
                        partition_scores,
                        all_generation_rewards,
                        raw_generation_rewards,
                    ) = await score_partition_all(
                        instance["prompt"],
                        instance["generations"],
                        instance["partition"],
                    )
                    utility = np.average(
                        generation_scores,
                        weights=patience ** np.arange(len(instance["generations"])),
                    )
                    return {
                        **instance,
                        "generation_scores": generation_scores,
                        "partition_scores": partition_scores,
                        "all_generation_rewards": all_generation_rewards,
                        "raw_generation_rewards": raw_generation_rewards,
                        "utility": utility,
                    }

            tasks = [process_single_instance(instance) for instance in instances]
            for result in tqdm(await asyncio.gather(*tasks), total=len(instances)):
                await f_scores.write(json.dumps(result) + "\n")
                generation_payload = {
                    "id": result["id"],
                    "model": result["model"],
                    "generation_rewards": result["all_generation_rewards"],
                    "raw_generation_rewards": result["raw_generation_rewards"],
                }
                await f_generations.write(json.dumps(generation_payload) + "\n")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    parser.add_argument(
        "--patience",
        help="Discount factor for computing cumulative utility.",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    instances = load_dataset(
        "json",
        data_files=os.path.join(eval_dir, "partitions.jsonl"),
        split="train",
    )

    os.makedirs(eval_dir, exist_ok=True)

    scores_file = os.path.join(eval_dir, "scores.jsonl")
    generation_rewards_file = os.path.join(eval_dir, "generation_rewards.jsonl")

    await process_instances(instances, scores_file, generation_rewards_file, args.patience)


if __name__ == "__main__":
    asyncio.run(main())
