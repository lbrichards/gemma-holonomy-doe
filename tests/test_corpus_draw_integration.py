import json
from pathlib import Path

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from corpus.draw import (
    BURNED_PILOT_DRAW_ORDERS,
    CONFIG_NAME,
    DATASET_NAME,
    N390_EXPERIMENT_SIZE,
    N390_TARGET_SURVIVORS,
    REVISION,
    TOKENIZER_NAME,
    DrawConfig,
    draw_stage_pool,
    partition_records_n390,
    partition_records,
)


@pytest.mark.integration
def test_full_seed42_draw_reproduces_committed_artifact():
    artifact = json.loads(Path("run/corpus_draw_stage_pool.json").read_text())
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, revision=REVISION, split="train")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    result = draw_stage_pool(rows=list(dataset), tokenizer=tokenizer, config=DrawConfig())
    records = artifact["records"]

    assert len(result.records) == 240
    assert [r.article_index for r in result.records] == [r["article_index"] for r in records]
    assert [r.token_ids for r in result.records] == [r["token_ids"] for r in records]

    partitions = partition_records(result.records)
    assert [r.draw_order for r in partitions["stage_1"]] == artifact["partitions"]["stage_1_draw_orders"]
    assert [r.article_index for r in partitions["stage_1"]] == artifact["partitions"]["stage_1_article_indices"]
    assert [r.draw_order for r in partitions["stage_2_reserve"]] == artifact["partitions"]["stage_2_reserve_draw_orders"]
    assert [r.article_index for r in partitions["stage_2_reserve"]] == artifact["partitions"]["stage_2_reserve_article_indices"]
    assert [r.draw_order for r in partitions["unused_reserve"]] == artifact["partitions"]["unused_reserve_draw_orders"]
    assert [r.article_index for r in partitions["unused_reserve"]] == artifact["partitions"]["unused_reserve_article_indices"]

    assert len(result.drops) == artifact["metadata"]["drop_count"] == len(artifact["drops"])
    assert [d.article_index for d in result.drops] == [d["article_index"] for d in artifact["drops"]]
    assert [d.reason for d in result.drops] == [d["reason"] for d in artifact["drops"]]


@pytest.mark.integration
def test_full_seed42_n390_draw_reproduces_committed_artifact_and_excludes_burned_points():
    artifact = json.loads(Path("run/corpus_draw_n390.json").read_text())
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, revision=REVISION, split="train")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    result = draw_stage_pool(
        rows=list(dataset),
        tokenizer=tokenizer,
        config=DrawConfig(target_survivors=N390_TARGET_SURVIVORS, burned_draw_orders=BURNED_PILOT_DRAW_ORDERS),
    )
    records = artifact["records"]
    burned = set(BURNED_PILOT_DRAW_ORDERS)

    assert len(result.records) == N390_TARGET_SURVIVORS
    assert artifact["metadata"]["survivor_count"] == N390_TARGET_SURVIVORS
    assert artifact["metadata"]["candidate_count"] == result.candidate_count
    assert artifact["metadata"]["excluded_burned_draw_orders"] == list(BURNED_PILOT_DRAW_ORDERS)
    assert [r.draw_order for r in result.burned_records] == list(BURNED_PILOT_DRAW_ORDERS)

    assert [r.article_index for r in result.records] == [r["article_index"] for r in records]
    assert [r.draw_order for r in result.records] == [r["draw_order"] for r in records]
    assert [r.token_ids for r in result.records] == [r["token_ids"] for r in records]
    assert [r.token_prefix_text for r in result.records] == [r["token_prefix_text"] for r in records]
    assert burned.isdisjoint({r.draw_order for r in result.records})
    assert burned.isdisjoint(set(artifact["partitions"]["experiment_draw_orders"]))
    assert burned.isdisjoint(set(artifact["partitions"]["reserve_draw_orders"]))

    partitions = partition_records_n390(result.records)
    assert len(partitions["experiment"]) == N390_EXPERIMENT_SIZE
    assert len(partitions["reserve"]) == N390_TARGET_SURVIVORS - N390_EXPERIMENT_SIZE
    assert [r.draw_order for r in partitions["experiment"]] == artifact["partitions"]["experiment_draw_orders"]
    assert [r.article_index for r in partitions["experiment"]] == artifact["partitions"]["experiment_article_indices"]
    assert [r.draw_order for r in partitions["reserve"]] == artifact["partitions"]["reserve_draw_orders"]
    assert [r.article_index for r in partitions["reserve"]] == artifact["partitions"]["reserve_article_indices"]

    assert len(result.drops) == artifact["metadata"]["drop_count"] == len(artifact["drops"])
    assert [d.article_index for d in result.drops] == [d["article_index"] for d in artifact["drops"]]
    assert [d.reason for d in result.drops] == [d["reason"] for d in artifact["drops"]]
