from __future__ import annotations

import random


def split_counts(total: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple[int, int, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
        if val_count == 0:
            val_count = 1
        test_count = total - train_count - val_count
        if test_count <= 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1
            test_count = total - train_count - val_count

    return train_count, val_count, test_count


def split_rows(
    rows: list[dict[str, object]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    train_count, val_count, _ = split_counts(len(rows), train_ratio=train_ratio, val_ratio=val_ratio)
    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)

    train_rows = shuffled_rows[:train_count]
    val_rows = shuffled_rows[train_count : train_count + val_count]
    test_rows = shuffled_rows[train_count + val_count :]
    return train_rows, val_rows, test_rows
