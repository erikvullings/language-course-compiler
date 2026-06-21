"""Vocabulary validator: tokenize → lemmatize → compare against allowed set.

Only *content words* (nouns, verbs, adjectives, adverbs) are validated;
function words (articles, conjunctions, prepositions, pronouns, etc.) pass
freely.  The caller supplies ``function_lemmas`` — typically all lemmas with a
function-word POS derived from the imported lexicon — so this module stays
language-agnostic.

Extra words beyond the allowed set are tolerated up to a configurable fraction
of the lesson's new-word count, provided they are at or below the target CEFR
level.  Words above the CEFR level (or with no CEFR tag) are always violations.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from course_compiler.generation.base import Lemmatizer

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)

# Canonical CEFR ordering — lower index = lower level.
CEFR_ORDER: list[str] = ["A1", "A2", "B1", "B2", "C1", "C2"]


def _cefr_exceeds(lemma_cefr: str | None, target: str) -> bool:
    """Return True if *lemma_cefr* is strictly above *target*, or unknown."""
    if lemma_cefr is None:
        return True  # conservative: unknown CEFR → violation
    try:
        return CEFR_ORDER.index(lemma_cefr) > CEFR_ORDER.index(target)
    except ValueError:
        return True


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single validation pass.

    ``violations`` must be empty for the lesson to be accepted.
    ``tolerated`` contains extra words that are within the CEFR/count budget
    (informational — not a reason to retry).
    """

    violations: frozenset[str]
    tolerated: frozenset[str]

    @property
    def is_valid(self) -> bool:
        return not self.violations


class VocabularyValidator:
    """Check that content words in a lesson belong to the allowed vocabulary.

    Args:
        lemmatizer: Maps surface tokens to their lemma (or ``None`` if unknown).
        function_lemmas: Lemmas always allowed (articles, prepositions, …).
    """

    def __init__(
        self,
        lemmatizer: Lemmatizer,
        function_lemmas: set[str] | None = None,
    ) -> None:
        self._lemmatizer = lemmatizer
        self._function_lemmas: set[str] = function_lemmas or set()

    def validate(
        self,
        text: str,
        allowed: set[str],
        *,
        extra_function_lemmas: set[str] | None = None,
        cefr_target: str | None = None,
        cefr_lookup: dict[str, str] | None = None,
        extra_tolerance: float | None = 0.5,
        new_word_count: int = 0,
    ) -> ValidationResult:
        """Validate *text* against *allowed* content-word lemmas.

        Args:
            text: Generated lesson text.
            allowed: Set of lemmas the lesson may freely use (prior + current words).
            extra_function_lemmas: Additional per-call exempt lemmas (e.g. verb forms).
            cefr_target: Target CEFR level (e.g. ``"A1"``).  Extra words above this
                level are always violations; words at or below are tolerated up to
                ``extra_tolerance * new_word_count``.
            cefr_lookup: ``{lemma: cefr_level}`` mapping derived from the lexicon.
            extra_tolerance: Fraction of ``new_word_count`` allowed as extras at or
                below the target CEFR (default 0.5 = 50 %). ``None`` means *no cap* —
                every at/below-CEFR extra is tolerated (coherence over strict
                prior-only discipline); above-CEFR words are still violations.
            new_word_count: Number of new words the lesson was supposed to introduce.
                Used to compute the tolerance budget.

        Returns:
            :class:`ValidationResult` — check ``.is_valid`` to decide whether to retry.
        """
        exempt = self._function_lemmas | (extra_function_lemmas or set())
        extras: set[str] = set()

        for raw_token in _TOKEN_RE.split(text):
            token = raw_token.strip()
            if not token:
                continue
            lemma = self._lemmatizer.lemmatize(token)
            resolved = lemma if lemma is not None else token.lower()
            if resolved in exempt or resolved in allowed:
                continue
            extras.add(resolved)

        if not extras:
            return ValidationResult(violations=frozenset(), tolerated=frozenset())

        # Classify extras by CEFR level when lookup is available.
        if cefr_target and cefr_lookup is not None:
            above_cefr = frozenset(e for e in extras if _cefr_exceeds(cefr_lookup.get(e), cefr_target))
            at_or_below = extras - above_cefr
        else:
            # No CEFR info: treat all extras as violations.
            return ValidationResult(violations=frozenset(extras), tolerated=frozenset())

        # Apply tolerance budget to at/below-CEFR extras. ``None`` = no cap: every
        # in-level extra is tolerated, so only above-CEFR words remain violations.
        if extra_tolerance is None:
            return ValidationResult(
                violations=above_cefr,
                tolerated=frozenset(at_or_below),
            )

        budget = math.ceil(extra_tolerance * max(new_word_count, 1))
        tolerated_list = sorted(at_or_below)[:budget]
        excess = frozenset(sorted(at_or_below)[budget:])

        return ValidationResult(
            violations=above_cefr | excess,
            tolerated=frozenset(tolerated_list),
        )
