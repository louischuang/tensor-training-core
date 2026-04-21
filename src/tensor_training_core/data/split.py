from __future__ import annotations

from collections.abc import Sequence


def split_counts(total: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def has_items(items: Sequence[object]) -> bool:
    return len(items) > 0
