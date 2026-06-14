import json
from pathlib import Path

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from corpus.draw import (
    CONFIG_NAME,
    DATASET_NAME,
    REVISION,
    TOKENIZER_NAME,
    DrawConfig,
    draw_stage_pool,
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
