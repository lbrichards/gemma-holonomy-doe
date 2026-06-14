#!/usr/bin/env python3
"""Acquire WikiText-103 raw via Hugging Face datasets.

Phase 26 Step 1 only. This script touches the network, verifies train split
plausibility, and stops. It does not sample base points or tokenize text.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


DATASET_NAME = "Salesforce/wikitext"
CONFIG_NAME = "wikitext-103-raw-v1"
REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MIN_PLAUSIBLE_TRAIN_EXAMPLES = 1_000_000


def _exception_headers(exc: BaseException) -> dict[str, Any] | None:
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return dict(headers)


def _print_exception(exc: BaseException) -> None:
    print("ACQUISITION: FAIL")
    print("Exception type:", type(exc).__name__)
    print("Exception message:", str(exc))
    headers = _exception_headers(exc)
    if headers:
        print("Exception response headers:")
        print(json.dumps(headers, indent=2, sort_keys=True))
        deny_headers = {
            key: value
            for key, value in headers.items()
            if "deny" in key.lower()
            or "reason" in key.lower()
            or "rate" in key.lower()
            or "x-error" in key.lower()
        }
        if deny_headers:
            print("Potential deny/reason headers:")
            print(json.dumps(deny_headers, indent=2, sort_keys=True))
    print("Full traceback:")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def _cache_paths(dataset: DatasetDict) -> list[str]:
    paths: list[str] = []
    for split_name, split in dataset.items():
        for cache_file in split.cache_files:
            filename = cache_file.get("filename")
            if filename:
                paths.append(f"{split_name}: {filename}")
    return paths


def main() -> int:
    print("WikiText-103 raw acquisition gate")
    print(f"dataset_name: {DATASET_NAME}")
    print(f"config_name: {CONFIG_NAME}")
    print(f"revision: {REVISION}")
    print(f"min_plausible_train_examples: {MIN_PLAUSIBLE_TRAIN_EXAMPLES}")
    try:
        api = HfApi()
        info = api.dataset_info(DATASET_NAME, revision=REVISION)
        print(f"hub_repo: {DATASET_NAME}")
        print(f"hub_revision_commit_sha: {info.sha}")
        print(f"revision_pinned_matches_requested: {info.sha == REVISION}")

        dataset = load_dataset(DATASET_NAME, CONFIG_NAME, revision=REVISION)
        split_sizes = {split: len(ds) for split, ds in dataset.items()}
        print("split_sizes:")
        print(json.dumps(split_sizes, indent=2, sort_keys=True))
        print("cache_files:")
        for path in _cache_paths(dataset):
            print(path)

        train_count = split_sizes.get("train")
        complete = train_count is not None and train_count >= MIN_PLAUSIBLE_TRAIN_EXAMPLES
        print(f"completeness_min_plausible_train: {MIN_PLAUSIBLE_TRAIN_EXAMPLES}")
        print(f"completeness_observed_train: {train_count}")
        print(f"completeness_nonzero: {bool(train_count)}")
        print(f"completeness_plausible: {complete}")
        print(f"completeness_check: {'PASS' if complete else 'FAIL'}")
        if complete:
            print("ACQUISITION: PASS")
            return 0
        print("ACQUISITION: FAIL")
        return 2
    except BaseException as exc:  # noqa: BLE001 - diagnostic gate prints full failure.
        _print_exception(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
