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

_WORD_RE = re.compile(r"\w+", re.UNICODE)

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

        for token_match in _WORD_RE.finditer(text):
            token = token_match.group(0)
            if not token:
                continue

            lemma = self._lemmatizer.lemmatize(token)
            resolved = lemma if lemma is not None else token.lower()

            # Proper names (e.g. "Mark") are usually acceptable discourse glue
            # and should not force retries. For sentence-initial titlecase words,
            # exemption is only applied when the resolved lemma is unknown to the
            # CEFR lookup (strong signal that it is a proper noun, not core vocab).
            if _looks_like_proper_name(token):
                if not _is_sentence_initial(text, token_match.start()):
                    continue
                if cefr_lookup is not None and resolved not in cefr_lookup:
                    continue

            if resolved in exempt or resolved in allowed:
                continue
            extras.add(resolved)

        if not extras:
            return ValidationResult(violations=frozenset(), tolerated=frozenset())

        # Classify extras by CEFR level when lookup is available.
        if cefr_target and cefr_lookup is not None:
            above_cefr = frozenset(
                e for e in extras if _cefr_exceeds(cefr_lookup.get(e), cefr_target)
            )
            at_or_below = extras - above_cefr
        else:
            # No CEFR info: treat all extras as violations.
            return ValidationResult(violations=frozenset(extras), tolerated=frozenset())

        # ``None`` tolerance = coherence-first generation: tolerate every extra the
        # lexicon does *not* know to be above level. Words with no CEFR tag (proper
        # nouns, rare/un-budgeted lemmas) are unknown, not provably advanced, so
        # rejecting them would sink otherwise-natural text; only words with a *known*
        # level strictly above the target remain violations.
        if extra_tolerance is None:
            known_above = frozenset(
                e
                for e in extras
                if (level := cefr_lookup.get(e)) is not None
                and _cefr_exceeds(level, cefr_target)
            )
            return ValidationResult(
                violations=known_above,
                tolerated=frozenset(extras - known_above),
            )

        budget = math.ceil(extra_tolerance * max(new_word_count, 1))
        tolerated_list = sorted(at_or_below)[:budget]
        excess = frozenset(sorted(at_or_below)[budget:])

        return ValidationResult(
            violations=above_cefr | excess,
            tolerated=frozenset(tolerated_list),
        )


def _looks_like_proper_name(token: str) -> bool:
    """Heuristic for personal names written in title case (e.g. ``Mark``)."""
    if len(token) < 2:
        return False
    if not token[0].isalpha() or not token[0].isupper():
        return False

    tail = token[1:]
    if not any(ch.islower() for ch in tail):
        return False

    for ch in tail:
        if ch in "-'":
            continue
        if not ch.isalpha():
            return False
    return True


def _is_sentence_initial(text: str, token_start: int) -> bool:
    """Return True when the token appears at a likely sentence start."""
    i = token_start - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    if i < 0:
        return True
    return text[i] in ".!?\n"
