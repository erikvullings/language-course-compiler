"""Reader for wordfreq's ``cBpack`` frequency files (language-agnostic).

A cBpack file is msgpack-encoded as ``[header, bucket_0, bucket_1, ...]`` where
``bucket_i`` lists the words whose frequency rounds to centibel ``-i`` (so lower
``i`` = more frequent). See https://github.com/rspeer/wordfreq. We expose a rank
(1-based, most frequent first) and a Zipf value per word.
"""

from __future__ import annotations

import math
from pathlib import Path

import msgpack

from course_compiler.models import Frequency

SOURCE = "wordfreq"


def _zipf_from_centibel(cb: int) -> float:
    """Zipf scale = log10(occurrences per billion). freq = 10**(-cb/100)."""

    return round(math.log10(10 ** (-cb / 100)) + 9, 2)


def load_frequencies(path: str | Path) -> dict[str, Frequency]:
    """Map each word to a :class:`Frequency` (rank + Zipf), most frequent first.

    The first occurrence of a word wins, so ranks are stable and unique.
    """

    with open(path, "rb") as fh:
        data = msgpack.load(fh, raw=False)

    if not data or not isinstance(data[0], dict) or data[0].get("format") != "cB":
        raise ValueError(f"{path} is not a wordfreq cBpack file")

    table: dict[str, Frequency] = {}
    rank = 0
    for cb, bucket in enumerate(data[1:], start=1):
        zipf = _zipf_from_centibel(cb)
        for word in bucket:
            if word in table:
                continue
            rank += 1
            table[word] = Frequency(rank=rank, zipf=zipf, source=SOURCE)
    return table
