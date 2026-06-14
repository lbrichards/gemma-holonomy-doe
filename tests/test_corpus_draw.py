from corpus.draw import (
    SEED_CORPUS,
    DrawConfig,
    draw_stage_pool,
    is_top_level_header,
    partition_records,
    reconstruct_articles,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [idx + 1 for idx, _ in enumerate(text.split())]


def article(title: str, n_words: int) -> list[str]:
    words = " ".join(f"{title.lower()}_{idx}" for idx in range(n_words))
    return [f" = {title} = ", words[: len(words) // 2], words[len(words) // 2 :]]


def synthetic_rows() -> list[str]:
    rows: list[str] = ["", "orphan preamble"]
    rows.extend(article("Alpha", 80))
    rows.extend([" = = Subsection = = ", "subsection text should stay with alpha"])
    rows.extend(article("Beta", 32))
    rows.extend(article("Gamma", 90))
    rows.extend(article("Delta", 100))
    rows.extend(article("Epsilon", 110))
    return rows


def test_header_detection_top_level_only():
    assert is_top_level_header(" = Article Title = ")
    assert is_top_level_header("= Compact Title =")
    assert not is_top_level_header(" = = Subtitle = = ")
    assert not is_top_level_header("== Subtitle ==")
    assert not is_top_level_header("plain prose line")


def test_reconstruct_articles_uses_top_level_headers_not_subheaders():
    articles = reconstruct_articles(synthetic_rows())

    assert [a.title for a in articles] == ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    assert "Subsection" in articles[0].text
    assert "subsection text should stay with alpha" in articles[0].text


def test_truncation_exactly_64_tokens_and_drops_logged():
    result = draw_stage_pool(
        rows=synthetic_rows(),
        tokenizer=FakeTokenizer(),
        config=DrawConfig(seed=SEED_CORPUS, target_survivors=3, raw_min_words=64, token_length=64),
    )

    assert len(result.records) == 3
    assert all(len(record.token_ids) == 64 for record in result.records)
    assert any(drop.reason == "raw_word_count_lt_64" and drop.article_title == "Beta" for drop in result.drops)


def test_draw_order_and_stage_partitions_are_recorded():
    result = draw_stage_pool(
        rows=synthetic_rows(),
        tokenizer=FakeTokenizer(),
        config=DrawConfig(seed=SEED_CORPUS, target_survivors=4, raw_min_words=64, token_length=64),
    )
    partitions = partition_records(result.records, stage_size=2)

    assert [record.draw_order for record in result.records] == [0, 1, 2, 3]
    assert len(partitions["stage_1"]) == 2
    assert len(partitions["stage_2_reserve"]) == 2
    assert len(partitions["unused_reserve"]) == 0


def test_determinism_same_seed_same_indices_and_tokens():
    config = DrawConfig(seed=SEED_CORPUS, target_survivors=4, raw_min_words=64, token_length=64)

    first = draw_stage_pool(rows=synthetic_rows(), tokenizer=FakeTokenizer(), config=config)
    second = draw_stage_pool(rows=synthetic_rows(), tokenizer=FakeTokenizer(), config=config)

    assert [r.article_index for r in first.records] == [r.article_index for r in second.records]
    assert [r.token_ids for r in first.records] == [r.token_ids for r in second.records]


def test_seed_sensitivity_changes_draw_order():
    first = draw_stage_pool(
        rows=synthetic_rows(),
        tokenizer=FakeTokenizer(),
        config=DrawConfig(seed=42, target_survivors=4, raw_min_words=64, token_length=64),
    )
    second = draw_stage_pool(
        rows=synthetic_rows(),
        tokenizer=FakeTokenizer(),
        config=DrawConfig(seed=7, target_survivors=4, raw_min_words=64, token_length=64),
    )

    assert [r.article_index for r in first.records] != [r.article_index for r in second.records]
