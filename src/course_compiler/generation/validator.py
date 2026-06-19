"""Vocabulary validator: tokenize → lemmatize → compare against allowed set.

Only *content words* (nouns, verbs, adjectives, adverbs) are validated;
function words (articles, conjunctions, prepositions, pronouns, etc.) pass
freely.  The caller supplies ``function_lemmas`` — typically the set of all
lemmas with a function-word POS derived from the imported lexicon — so this
module stays language-agnostic.
"""

from __future__ import annotations

import re

from course_compiler.generation.base import Lemmatizer

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)


class VocabularyValidator:
    """Check that content words in a lesson belong to the allowed vocabulary.

    Args:
        lemmatizer: Maps surface tokens to their lemma (or ``None`` if unknown).
        function_lemmas: Lemmas that are always allowed (articles, prepositions,
            conjunctions, pronouns, …).  Tokens whose lemma is in this set are
            skipped during validation.  Pass an empty set to validate every token.
    """

    def __init__(
        self,
        lemmatizer: Lemmatizer,
        function_lemmas: set[str] | None = None,
    ) -> None:
        self._lemmatizer = lemmatizer
        self._function_lemmas: set[str] = function_lemmas or set()

    def validate(self, text: str, allowed: set[str]) -> set[str]:
        """Return the set of unknown content-word lemmas found in *text*.

        An empty return value means the text passes validation.
        Tokens whose lemma is in ``function_lemmas`` are not checked.
        Tokens the lemmatizer cannot resolve are reported by their lowercase
        surface form so the caller can inspect what the LLM introduced.
        """
        unknown: set[str] = set()
        for raw_token in _TOKEN_RE.split(text):
            token = raw_token.strip()
            if not token:
                continue
            lemma = self._lemmatizer.lemmatize(token)
            resolved = lemma if lemma is not None else token.lower()
            if resolved in self._function_lemmas:
                continue
            if resolved not in allowed:
                unknown.add(resolved)
        return unknown
