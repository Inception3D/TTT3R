from __future__ import annotations

import math
from typing import Sequence


def uniform_sample_indices(total: int, target: int) -> list[int]:
    if total <= 0:
        return []
    if target <= 0:
        return []
    if target >= total:
        return list(range(total))
    if target == 1:
        return [0]
    step = (total - 1) / (target - 1)
    indices = [int(round(i * step)) for i in range(target)]
    # Guard duplicates due to rounding in very short sequences.
    deduped = []
    last = -1
    for idx in indices:
        idx = max(0, min(total - 1, idx))
        if idx <= last:
            idx = min(total - 1, last + 1)
        deduped.append(idx)
        last = idx
    return deduped


def parse_length_from_seq_name(name: str) -> int | None:
    marker = "_len"
    if marker not in name:
        return None
    tail = name.split(marker, 1)[1]
    digits = []
    for c in tail:
        if c.isdigit():
            digits.append(c)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))
