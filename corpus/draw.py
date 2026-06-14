#!/usr/bin/env python3
"""Deterministic WikiText article draw for the Gemma holonomy DOE.

This is corpus construction only: no activations, no planes, no holonomy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "Salesforce/wikitext"
CONFIG_NAME = "wikitext-103-raw-v1"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TOKENIZER_NAME = "google/gemma-2-2b"
SEED_CORPUS = 42
TARGET_SURVIVORS = 240
TOKEN_LENGTH = 64
RAW_MIN_WORDS = 64
OUTPUT_PATH = Path("run/corpus_draw_stage_pool.json")
N390_OUTPUT_PATH = Path("run/corpus_draw_n390.json")
BURNED_PILOT_DRAW_ORDERS = tuple(range(96, 112))
N390_TARGET_SURVIVORS = 700
N390_EXPERIMENT_SIZE = 390

_TOP_LEVEL_HEADER = re.compile(r"^\s*=\s+([^=].*?[^=])\s+=\s*$")


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True)
class DrawConfig:
    seed: int = SEED_CORPUS
    target_survivors: int = TARGET_SURVIVORS
    raw_min_words: int = RAW_MIN_WORDS
    token_length: int = TOKEN_LENGTH
    burned_draw_orders: tuple[int, ...] = ()


@dataclass(frozen=True)
class Article:
    article_index: int
    title: str
    start_row: int
    end_row: int
    text: str
    raw_word_count: int


@dataclass(frozen=True)
class DrawRecord:
    draw_order: int
    article_index: int
    article_title: str
    start_row: int
    end_row: int
    token_ids: list[int]
    token_count_before_truncation: int
    token_prefix_text: str


@dataclass(frozen=True)
class DropRecord:
    article_index: int
    article_title: str
    reason: str
    raw_word_count: int
    token_count: int | None = None


@dataclass(frozen=True)
class DrawResult:
    records: list[DrawRecord]
    drops: list[DropRecord]
    total_articles: int
    raw_prefilter_survivors: int
    seed: int
    target_survivors: int
    token_length: int
    candidate_count: int
    burned_draw_orders: tuple[int, ...]
    burned_records: list[DrawRecord]


def is_top_level_header(line: str) -> bool:
    """Return true for WikiText top-level article headers, not subheaders."""

    stripped = line.strip()
    if stripped.startswith("==") or stripped.startswith("= ="):
        return False
    return _TOP_LEVEL_HEADER.match(line) is not None


def _header_title(line: str) -> str:
    match = _TOP_LEVEL_HEADER.match(line)
    if match is None:
        raise ValueError(f"not a top-level header: {line!r}")
    return " ".join(match.group(1).split())


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict) and "text" in row:
        return str(row["text"])
    return str(row)


def reconstruct_articles(rows: list[Any]) -> list[Article]:
    """Reconstruct WikiText articles from line-level rows and top-level headers."""

    articles: list[Article] = []
    current_title: str | None = None
    current_start: int | None = None
    current_lines: list[str] = []

    def flush(end_row: int) -> None:
        nonlocal current_title, current_start, current_lines
        if current_title is None or current_start is None:
            return
        text = "\n".join(line.strip() for line in current_lines if line.strip())
        articles.append(
            Article(
                article_index=len(articles),
                title=current_title,
                start_row=current_start,
                end_row=end_row,
                text=text,
                raw_word_count=len(text.split()),
            )
        )

    for row_index, row in enumerate(rows):
        line = _row_text(row)
        if is_top_level_header(line):
            flush(row_index - 1)
            current_title = _header_title(line)
            current_start = row_index
            current_lines = [line]
            continue
        if current_title is not None:
            current_lines.append(line)

    flush(len(rows) - 1)
    return articles


def partition_records(records: list[DrawRecord], stage_size: int = 96) -> dict[str, list[DrawRecord]]:
    """Return the Section 9 first-96 / next-96 / remainder draw partition."""

    return {
        "stage_1": records[:stage_size],
        "stage_2_reserve": records[stage_size : 2 * stage_size],
        "unused_reserve": records[2 * stage_size :],
    }


def partition_records_n390(records: list[DrawRecord], experiment_size: int = N390_EXPERIMENT_SIZE) -> dict[str, list[DrawRecord]]:
    """Return the v2 single-stage experiment/reserve partition."""

    return {
        "experiment": records[:experiment_size],
        "reserve": records[experiment_size:],
    }


def draw_stage_pool(*, rows: list[Any], tokenizer: TokenizerLike, config: DrawConfig = DrawConfig()) -> DrawResult:
    """Draw deterministic reconstructed articles and truncate to Gemma tokens."""

    articles = reconstruct_articles(rows)
    drops: list[DropRecord] = []
    eligible: list[Article] = []
    for article in articles:
        if article.raw_word_count < config.raw_min_words:
            drops.append(
                DropRecord(
                    article_index=article.article_index,
                    article_title=article.title,
                    reason="raw_word_count_lt_64",
                    raw_word_count=article.raw_word_count,
                )
            )
        else:
            eligible.append(article)

    rng = random.Random(config.seed)
    shuffled = list(eligible)
    rng.shuffle(shuffled)

    records: list[DrawRecord] = []
    burned_records: list[DrawRecord] = []
    burned_draw_orders = set(config.burned_draw_orders)
    survivor_draw_order = 0
    candidate_count = 0
    for article in shuffled:
        if len(records) >= config.target_survivors:
            break
        candidate_count += 1
        token_ids = tokenizer.encode(article.text, add_special_tokens=False)
        if len(token_ids) < config.token_length:
            drops.append(
                DropRecord(
                    article_index=article.article_index,
                    article_title=article.title,
                    reason="token_count_lt_64",
                    raw_word_count=article.raw_word_count,
                    token_count=len(token_ids),
                )
            )
            continue
        truncated = token_ids[: config.token_length]
        record = DrawRecord(
            draw_order=survivor_draw_order,
            article_index=article.article_index,
            article_title=article.title,
            start_row=article.start_row,
            end_row=article.end_row,
            token_ids=truncated,
            token_count_before_truncation=len(token_ids),
            token_prefix_text=tokenizer.decode(truncated),
        )
        survivor_draw_order += 1
        if record.draw_order in burned_draw_orders:
            burned_records.append(record)
            continue
        records.append(record)

    return DrawResult(
        records=records,
        drops=drops,
        total_articles=len(articles),
        raw_prefilter_survivors=len(eligible),
        seed=config.seed,
        target_survivors=config.target_survivors,
        token_length=config.token_length,
        candidate_count=candidate_count,
        burned_draw_orders=tuple(config.burned_draw_orders),
        burned_records=burned_records,
    )


def _result_to_json(result: DrawResult) -> dict[str, Any]:
    partitions = partition_records(result.records)
    return {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "config_name": CONFIG_NAME,
            "revision": REVISION,
            "tokenizer_name": TOKENIZER_NAME,
            "seed": result.seed,
            "target_survivors": result.target_survivors,
            "token_length": result.token_length,
            "total_articles": result.total_articles,
            "raw_prefilter_survivors": result.raw_prefilter_survivors,
            "survivor_count": len(result.records),
            "drop_count": len(result.drops),
            "candidate_count": result.candidate_count,
            "excluded_burned_draw_orders": list(result.burned_draw_orders),
            "excluded_burned_article_indices": [record.article_index for record in result.burned_records],
        },
        "partitions": {
            "stage_1_draw_orders": [record.draw_order for record in partitions["stage_1"]],
            "stage_1_article_indices": [record.article_index for record in partitions["stage_1"]],
            "stage_2_reserve_draw_orders": [record.draw_order for record in partitions["stage_2_reserve"]],
            "stage_2_reserve_article_indices": [record.article_index for record in partitions["stage_2_reserve"]],
            "unused_reserve_draw_orders": [record.draw_order for record in partitions["unused_reserve"]],
            "unused_reserve_article_indices": [record.article_index for record in partitions["unused_reserve"]],
        },
        "records": [asdict(record) for record in result.records],
        "drops": [asdict(drop) for drop in result.drops],
    }


def _result_to_json_n390(result: DrawResult, experiment_size: int = N390_EXPERIMENT_SIZE) -> dict[str, Any]:
    partitions = partition_records_n390(result.records, experiment_size=experiment_size)
    return {
        "metadata": {
            "dataset_name": DATASET_NAME,
            "config_name": CONFIG_NAME,
            "revision": REVISION,
            "tokenizer_name": TOKENIZER_NAME,
            "seed": result.seed,
            "target_survivors": result.target_survivors,
            "token_length": result.token_length,
            "total_articles": result.total_articles,
            "raw_prefilter_survivors": result.raw_prefilter_survivors,
            "survivor_count": len(result.records),
            "drop_count": len(result.drops),
            "candidate_count": result.candidate_count,
            "excluded_burned_draw_orders": list(result.burned_draw_orders),
            "excluded_burned_article_indices": [record.article_index for record in result.burned_records],
            "excluded_burned_records": [asdict(record) for record in result.burned_records],
            "experiment_size": experiment_size,
            "reserve_count": len(partitions["reserve"]),
            "partition": "single_stage_experiment_plus_reserve",
        },
        "partitions": {
            "experiment_draw_orders": [record.draw_order for record in partitions["experiment"]],
            "experiment_article_indices": [record.article_index for record in partitions["experiment"]],
            "reserve_draw_orders": [record.draw_order for record in partitions["reserve"]],
            "reserve_article_indices": [record.article_index for record in partitions["reserve"]],
        },
        "records": [asdict(record) for record in result.records],
        "drops": [asdict(drop) for drop in result.drops],
    }


def write_stage_pool(result: DrawResult, path: Path = OUTPUT_PATH) -> None:
    """Write the blind corpus draw artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_result_to_json(result), indent=2, sort_keys=True) + "\n")


def write_n390_pool(result: DrawResult, path: Path = N390_OUTPUT_PATH) -> None:
    """Write the v2 N=390 blind corpus draw artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_result_to_json_n390(result), indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--target-survivors", type=int, default=TARGET_SURVIVORS)
    parser.add_argument("--n390", action="store_true", help="Write the v2 N=390 artifact with burned pilot points excluded.")
    args = parser.parse_args()

    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, revision=REVISION, split="train")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    config = DrawConfig(target_survivors=args.target_survivors)
    if args.n390:
        config = DrawConfig(target_survivors=args.target_survivors, burned_draw_orders=BURNED_PILOT_DRAW_ORDERS)
        if args.output == OUTPUT_PATH:
            args.output = N390_OUTPUT_PATH
    result = draw_stage_pool(
        rows=list(dataset),
        tokenizer=tokenizer,
        config=config,
    )
    if args.n390:
        write_n390_pool(result, args.output)
        partitions = partition_records_n390(result.records)
    else:
        write_stage_pool(result, args.output)
        partitions = partition_records(result.records)
    print(f"dataset_revision: {REVISION}")
    print(f"total_articles: {result.total_articles}")
    print(f"raw_prefilter_survivors: {result.raw_prefilter_survivors}")
    print(f"candidate_count: {result.candidate_count}")
    print(f"survivor_count: {len(result.records)}")
    print(f"drop_count: {len(result.drops)}")
    if args.n390:
        print(f"excluded_burned_draw_orders: {list(result.burned_draw_orders)}")
        print(f"excluded_burned_count: {len(result.burned_records)}")
        print(f"experiment_count: {len(partitions['experiment'])}")
        print(f"reserve_count: {len(partitions['reserve'])}")
    else:
        print(f"stage_1_count: {len(partitions['stage_1'])}")
        print(f"stage_2_reserve_count: {len(partitions['stage_2_reserve'])}")
        print(f"unused_reserve_count: {len(partitions['unused_reserve'])}")
    print(f"output: {args.output}")
    if len(result.records) < args.target_survivors:
        print("DRAW: FAIL")
        return 2
    print("DRAW: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
