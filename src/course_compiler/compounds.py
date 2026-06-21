"""Compound / derivable-word detection (language-pluggable).

Many languages (Dutch, German, …) form transparent compounds by concatenating
known words, sometimes joined by a *linking morpheme*: ``koffie`` + ``pot`` →
``koffiepot``; ``station`` + ``s`` + ``plein`` → ``stationsplein``. A learner who
knows the parts gets the compound nearly for free, so such words may be
*introduced* in a lesson but should not consume the frequency budget (cf. task
0017/0018).

This module decides whether a word decomposes into ≥2 known parts. It is generic:
the linking morphemes are passed in by the caller (the language-specific
converter), and ``models.py`` stays untouched. Opaque compounds whose meaning is
not the sum of the parts (``handschoen`` = hand-shoe = glove) are *not* free; the
caller marks them via ``opaque`` so they keep counting as new words.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def build_known_parts(
    known_lemmas: Collection[str], min_part_len: int = 2
) -> frozenset[str]:
    """Pre-filter the candidate-part set once for reuse across many splits.

    Building this per call is the dominant cost when checking a whole lexicon
    (tens of thousands of words against a same-size set), so callers in a hot loop
    should build it once and pass it to :func:`split_with_known` /
    :func:`is_derivable_with_known`.
    """
    return frozenset(k for k in known_lemmas if len(k) >= min_part_len)


def split_with_known(
    word: str,
    known_parts: frozenset[str],
    *,
    linkers: Sequence[str] = (),
    min_part_len: int = 2,
) -> list[str]:
    """Split *word* against a pre-built part set (see :func:`build_known_parts`)."""

    def segment(rest: str) -> list[str] | None:
        # Prefer the longest leading known part for a stable, greedy result.
        for length in range(len(rest), min_part_len - 1, -1):
            part = rest[:length]
            # ``part == word`` skips the whole-word match so a word in its own set
            # isn't "split" into a single copy of itself (must be ≥2 real parts).
            if part == word or part not in known_parts:
                continue
            tail = rest[length:]
            if not tail:
                return [part]
            # First try an adjacent next part, then allow a linking morpheme.
            sub = segment(tail)
            if sub is not None:
                return [part, *sub]
            for link in linkers:
                if link and tail.startswith(link):
                    sub = segment(tail[len(link) :])
                    if sub is not None:
                        return [part, *sub]
        return None

    result = segment(word)
    return result if result is not None and len(result) >= 2 else []


def is_derivable_with_known(
    word: str,
    known_parts: frozenset[str],
    *,
    linkers: Sequence[str] = (),
    opaque: Collection[str] = (),
    min_part_len: int = 2,
) -> bool:
    """Transparent-compound check against a pre-built part set."""
    if word in opaque:
        return False
    return (
        len(split_with_known(word, known_parts, linkers=linkers, min_part_len=min_part_len))
        >= 2
    )


def split_compound(
    word: str,
    known_lemmas: Collection[str],
    *,
    linkers: Sequence[str] = (),
    min_part_len: int = 2,
) -> list[str]:
    """Split *word* into known parts, or return ``[]`` if it doesn't decompose.

    Convenience wrapper that builds the part set per call — fine for one-off use;
    use :func:`build_known_parts` + :func:`split_with_known` in a hot loop.

    Args:
        word: The candidate compound.
        known_lemmas: Lemmas that may serve as parts. The word itself is never used
            as a single part, so a word is never "split" into one copy of itself.
        linkers: Optional linking morphemes allowed *between* parts (e.g. ``"s"``,
            ``"en"``). Tried in order; first successful split wins (deterministic).
        min_part_len: Minimum length of each part, to suppress tiny-fragment false
            positives.

    Returns:
        The list of parts (length ≥ 2) for the first valid segmentation found,
        preferring longer leading parts, or ``[]`` when no segmentation exists.
    """
    known = build_known_parts(known_lemmas, min_part_len)
    return split_with_known(word, known, linkers=linkers, min_part_len=min_part_len)


def is_derivable_compound(
    word: str,
    known_lemmas: Collection[str],
    *,
    linkers: Sequence[str] = (),
    opaque: Collection[str] = (),
    min_part_len: int = 2,
) -> bool:
    """True if *word* is a *transparent* compound of ≥2 known parts.

    Words listed in ``opaque`` are treated as non-transparent (they still count as
    new words even though they decompose).
    """
    if word in opaque:
        return False
    return (
        len(split_compound(word, known_lemmas, linkers=linkers, min_part_len=min_part_len))
        >= 2
    )
