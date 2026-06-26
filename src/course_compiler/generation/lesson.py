"""LessonGenerator: build lesson content via LLM with vocabulary validation + retry."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Literal

from course_compiler.generation.base import Lemmatizer
from course_compiler.generation.cache import LLMCache
from course_compiler.generation.validator import ValidationResult, VocabularyValidator
from course_compiler.llm.base import LLMError, LLMProvider, Message, Role

#: Words of lesson text generated per new content word introduced.
WORDS_PER_NEW_WORD = 15

#: Level-specific structural and tense constraints for CEFR A1-B2
LEVEL_CONSTRAINTS = {
    "A1": {
        "sentence_structure": "Simple, short, declarative sentences. Avoid complex subordinate clauses, relative clauses, or passive voice.",
        "tenses": "Exclusively Present Tense. Simple past of highly frequent copula verbs (e.g., 'was/were') is allowed only if completely necessary for natural flow.",
    },
    "A2": {
        "sentence_structure": "Simple sentences linked with basic connectors like 'and', 'but', 'because'. Direct subordination is allowed, but keep syntax clean.",
        "tenses": "Primarily Present Tense, but basic Past Tenses (Perfect/Imperfect) should be integrated naturally where narrative requires.",
    },
    "B1": {
        "sentence_structure": "A mix of simple and complex sentences. Can link narratives smoothly using various coordinating and subordinating conjunctions.",
        "tenses": "Full range of narrative tenses (Present, Past, Perfect). Future and basic conditional expressions can be used naturally.",
    },
    "B2": {
        "sentence_structure": "Complex, fluid sentence structures showing clear control of syntax, sub-clauses, and stylistic variation.",
        "tenses": "Advanced narrative tenses, including passives, conditionals, and subjunctive forms if native to the target language.",
    },
}


def _target_length(new_word_count: int, cefr: str = "A1") -> tuple[int, int]:
    """Return (min_words, max_words) for lesson text based on vocabulary and CEFR level.

    Scales minimum length by CEFR level to ensure richer, more natural narratives
    at higher proficiency levels.
    """
    level_min_factor = {"A1": 60, "A2": 75, "B1": 90, "B2": 105}.get(cefr, 60)
    min_words = max(level_min_factor, 60)
    max_words = max(new_word_count * WORDS_PER_NEW_WORD, min_words + 30)
    return (min_words, max_words)


@dataclass(frozen=True)
class AttemptDiagnostics:
    attempt: int
    violations: frozenset[str]
    tolerated: frozenset[str]


@dataclass(frozen=True)
class GeneratedLesson:
    lesson_id: str
    content: str
    attempts: int
    title: str | None = None
    tolerated: frozenset[str] = frozenset()
    fallback: bool = False
    violations: frozenset[str] = frozenset()
    best_attempt: int | None = None
    diagnostics: tuple[AttemptDiagnostics, ...] = ()


_LOW_CEFR: frozenset[str] = frozenset({"A1", "A2"})
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TITLE_TEXT_RE = re.compile(
    r"(?is)^\s*TITLE:\s*(.*?)\s*^\s*TEXT:\s*(.*)$", re.MULTILINE
)


def _language_specific_checklist(language: str) -> str:
    """Return language-specific grammar checklist for quality control.

    Each language has unique structural and agreement rules that should be
    verified before returning the final lesson text.
    """
    checklists = {
        "nl": (
            "1. Subject-Verb Agreement: Ensure absolute agreement for all pronouns "
            "(e.g., 'ik ben', 'jij bent', 'hij is', 'wij zijn').\n"
            "2. V2 Word Order in Main Clauses: The finite verb must be the second element. "
            "In subordinate clauses, the verb moves to the end.\n"
            "3. Adjective Inflection: Apply correct gender and number agreement to adjectives "
            "when used attributively (e.g., 'een mooie stad').\n"
            "4. Contextual Semantic Precision: Distinguish between related verbs "
            "(e.g., 'wonen' = to live/reside vs. 'leven' = to live/exist).\n"
            "5. Completeness: Every sentence must have a clear, complete structure "
            "appropriate for the target CEFR level."
        ),
        "de": (
            "1. Subject-Verb-Object Word Order: In main clauses, follow SVO order. "
            "In subordinate clauses, the verb moves to the end.\n"
            "2. Case Agreement: Ensure correct nominative, accusative, dative, and genitive cases "
            "for articles, adjectives, and pronouns.\n"
            "3. Gender-Number Agreement: Verify gender/number agreement between "
            "nouns, articles, and adjectives.\n"
            "4. Separable Verbs: Ensure correct separation in main clauses "
            "(e.g., 'Ich rufe dich an').\n"
            "5. Completeness: Every sentence must be structurally complete and appropriate "
            "for the target CEFR level."
        ),
        "fr": (
            "1. Subject-Verb Agreement: Verify conjugation matches the subject pronoun "
            "(e.g., 'je suis', 'tu es', 'il est').\n"
            "2. Gender-Number Harmony: Adjectives and articles must agree in gender and number "
            "with their nouns.\n"
            "3. Verb Placement: Main clause verbs typically follow the subject. "
            "Subordinate clauses follow the same SVO order.\n"
            "4. Reflexive Accuracy: When using reflexive verbs, ensure the correct reflexive "
            "pronoun and word order (e.g., 'Je me lève').\n"
            "5. Completeness: Every sentence must be clear, complete, and appropriate "
            "for the target CEFR level."
        ),
    }
    # Return language-specific checklist or a generic one if language not defined
    return checklists.get(
        language.lower()[:2],
        (
            "1. Verb Conjugation: Ensure verbs are conjugated to match their subject.\n"
            "2. Agreement: Apply correct agreement rules for articles, adjectives, and pronouns.\n"
            "3. Word Order: Follow target language word order conventions.\n"
            "4. Completeness: Every sentence must be structurally complete and clear.\n"
            "5. Coherence: The narrative should flow naturally and make sense."
        ),
    )


def _user_instructions(
    language: str,
    min_words: int,
    max_words: int,
    cefr: str = "A1",
) -> str:
    """Generate dynamic, level-aware instructions for lesson writing.

    Args:
        language: Target language (e.g., 'Dutch', 'German', 'French').
        min_words: Minimum target lesson length.
        max_words: Maximum target lesson length.
        cefr: CEFR level ('A1', 'A2', 'B1', 'B2').

    Returns:
        A parameterized prompt template ready for format() with theme, goals, etc.
    """
    constraints = LEVEL_CONSTRAINTS.get(cefr, LEVEL_CONSTRAINTS["A1"])
    checklist = _language_specific_checklist(language)
    markdown_formatting_criteria = (
        "   * Use italics (*text*) exclusively for spoken dialogue lines.\n"
        "   * Use bold (**text**) exclusively for imperative verbs and strong command formats.\n"
        "   * Do not bold or italicize full narrative sentences or use formatting for general emphasis.\n"
    )

    return (
        f"You are an expert CEFR language course curriculum writer and a native speaker of {language}. "
        f"Your task is to write a highly natural, grammatically flawless short story tailored precisely to the parameters provided below.\n\n"
        f"### Core Constraints\n"
        f"1. Length: Minimum {min_words} words, Maximum {max_words} words.\n"
        f"2. CEFR Level: Strict {cefr}. Structure sentences according to these criteria: {constraints['sentence_structure']}\n"
        f"3. Tense: Apply these tense constraints: {constraints['tenses']}\n"
        f"4. Formatting: Output ONLY a raw JSON object with exactly two keys: "
        f'{{{{"title": "<title>", "text": "<markdown_text>"}}}}. '
        f"Do not wrap the JSON in markdown code fences (```json ... ```). No pre- or post-commentary.\n"
        f"5. Markdown inside JSON: Use plain paragraphs. No bold or italicized full sentences.\n"
        f"{markdown_formatting_criteria}"
        f"6. Title Style: A noun phrase of exactly 2 to 6 words in {language}. "
        f"It must not be a question or a sentence fragment.\n\n"
        f"### Target Content Parameters\n"
        f"- Theme: {{theme}}\n"
        f"- Communicative Goals:\n{{communicative_goals_list}}\n"
        f"- Preferred Vocabulary / Concepts:\n"
        f"  * Key Words: {{seed_words}}\n"
        f"  * Target Verbs: {{verbs}}\n"
        f"  * (Note: Prioritize weaving these words/concepts naturally into the context of the story. "
        f"They are preferred guidelines rather than hard, robotic insertions.)\n\n"
        f"### Target Language Quality Control Checklist\n"
        f"Before rendering the JSON, run a multi-pass internal verification to ensure perfect {language} grammar and syntax:\n"
        f"{checklist}\n\n"
        f"### Story Outline\n{{outline}}\n\n"
        f"Generate the raw JSON response now:"
    )


def _extract_json_payload(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if not text:
        return None
    candidates = [text]
    fenced = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        candidates.append(fenced.group(1).strip())
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _extract_labeled_payload(raw: str) -> tuple[str | None, str] | None:
    match = _TITLE_TEXT_RE.search(raw.strip())
    if match is None:
        return None
    title = match.group(1).strip()
    text = match.group(2).strip()
    if not text:
        return None
    return (title or None), text


def _shorten_title(title: str, fallback_text: str) -> str:
    base = title.strip()
    if not base:
        for line in fallback_text.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                base = stripped
                break

    # Remove trailing punctuation
    while base and base[-1] in ".!?,;:":
        base = base[:-1].strip()

    words = [w for w in base.split() if w]

    # If title is already suspiciously long (looks like a full sentence or paragraph),
    # take only the first few words as a conservative estimate.
    if len(words) > 8:
        base = " ".join(words[:4])  # Reduce to first 4 words for long inputs
    elif len(words) > 6:
        base = " ".join(words[:6])

    return base.strip()


def _extract_title_and_text(raw: str) -> tuple[str | None, str]:
    labeled = _extract_labeled_payload(raw)
    if labeled is not None:
        title, text = labeled
        return _shorten_title(title or "", text) or None, text

    payload = _extract_json_payload(raw)
    if payload is None:
        return None, raw

    text_value = payload.get("text")
    text = text_value.strip() if isinstance(text_value, str) else ""
    if not text:
        return None, raw

    title_value = payload.get("title")
    title = title_value.strip() if isinstance(title_value, str) else ""
    return _shorten_title(title, text) or None, text


def _feedback_message(result: ValidationResult, cefr_target: str) -> str:
    parts: list[str] = [
        "Your previous response contained vocabulary problems. Please rewrite the lesson."
    ]
    if result.violations:
        words = ", ".join(sorted(result.violations))
        parts.append(
            f"These words must not appear — they are above {cefr_target} level "
            f"or outside the allowed vocabulary: {words}."
        )
    return " ".join(parts)


def _fallback_content(new_words: list[str], *, lesson_id: str, reason: str) -> str:
    """Return deterministic placeholder lesson content when the provider fails."""
    words = " ".join(new_words).strip()
    if words:
        return f"{words}."
    return f"{lesson_id}: lesson unavailable ({reason})."


class LessonGenerator:
    """Generate lesson text via an :class:`~course_compiler.llm.base.LLMProvider`.

    Validates content words against the allowed vocabulary and samples multiple
    drafts, selecting the one with the fewest violations.

    Extra words at or below the target CEFR level are tolerated up to
    ``extra_tolerance`` × ``len(new_words)`` (default 50 %).  Words above the
    target CEFR are always rejected.

    Optionally caches LLM responses (first-attempt only) for reproducibility.
    """

    def __init__(
        self,
        provider: LLMProvider,
        lemmatizer: Lemmatizer,
        *,
        function_lemmas: set[str] | None = None,
        cache: LLMCache | None = None,
        max_retries: int = 5,
        extra_tolerance: float | None = 0.5,
        retry_strategy: Literal["natural", "corrective"] = "natural",
        fail_open_on_llm_error: bool = True,
        fail_open_on_validation_error: bool = True,
        verbose: bool = False,
    ) -> None:
        self._provider = provider
        self._validator = VocabularyValidator(lemmatizer, function_lemmas)
        self._cache = cache
        self._max_retries = max_retries
        self._extra_tolerance = extra_tolerance
        self._retry_strategy: Literal["natural", "corrective"] = retry_strategy
        self._fail_open_on_llm_error = fail_open_on_llm_error
        self._fail_open_on_validation_error = fail_open_on_validation_error
        self._verbose = verbose

    def _log_messages(self, attempt: int, messages: list[Message]) -> None:
        if not self._verbose:
            return
        print(
            f"[llm][lesson][attempt {attempt}] prompt follows",
            file=sys.stderr,
        )
        for index, message in enumerate(messages, start=1):
            print(
                f"--- message {index} ({message.role.value}) ---",
                file=sys.stderr,
            )
            print(message.content, file=sys.stderr)

    def _build_initial_messages(
        self,
        lesson_id: str,
        language: str,
        cefr: str,
        theme: str,
        new_words: list[str],
        min_words: int,
        max_words: int,
        *,
        outline: str = "",
        communicative_goals: list[str] | None = None,
        glosses: dict[str, str] | None = None,
        verb_lemmas: list[str] | None = None,
        english_seed_words: list[str] | None = None,
    ) -> list[Message]:
        """Build the initial LLM prompt with dynamic level-appropriate constraints."""
        # Format lesson vocabulary for display
        rendered_words = [
            (
                f"{lemma} ({glosses[lemma]})"
                if glosses is not None and lemma in glosses and glosses[lemma]
                else lemma
            )
            for lemma in new_words
        ]

        # Format communicative goals for display
        goals = communicative_goals or []
        goals_str = (
            "\n".join(f"  - {goal}" for goal in goals)
            if goals
            else "  - (None specified)"
        )

        # Format verbs for display
        verbs = verb_lemmas or []
        verbs_str = ", ".join(verbs) if verbs else "(None)"

        # Format seed words for display
        seeds = english_seed_words or []
        seeds_str = ", ".join(seeds) if seeds else "(None)"

        # Get the template and format it with all parameters
        prompt_template = _user_instructions(language, min_words, max_words, cefr)
        user_content = prompt_template.format(
            theme=theme,
            communicative_goals_list=goals_str,
            seed_words=seeds_str,
            verbs=verbs_str,
            outline=outline or "(No outline provided)",
        )

        return [Message(Role.USER, user_content)]

    def generate(
        self,
        lesson_id: str,
        new_words: list[str],
        allowed_words: set[str],
        *,
        language: str,
        cefr: str = "A1",
        theme: str = "general",
        model: str | None = None,
        temperature: float = 0.7,
        function_lemmas: set[str] | None = None,
        cefr_lookup: dict[str, str] | None = None,
        outline: str = "",
        communicative_goals: list[str] | None = None,
        glosses: dict[str, str] | None = None,
        verb_lemmas: list[str] | None = None,
        english_seed_words: list[str] | None = None,
    ) -> GeneratedLesson:
        """Generate and validate lesson content, sampling multiple drafts.

        Args:
            lesson_id: Unique identifier for this lesson (e.g. ``"lesson001"``).
            new_words: Content-word lemmas being introduced for the first time.
            allowed_words: Full set of content-word lemmas the validator accepts.
                Not sent to the LLM — enforced by the validator only.
            language: Human-readable target language name (e.g. ``"Dutch"``).
            cefr: CEFR level (e.g. ``"A1"``). Included in the prompt and used to
                classify extra words as tolerated vs. violations.
            theme: Semantic theme (e.g. ``"home"``). Included in the prompt.
            model: LLM model identifier; provider default used if omitted.
            temperature: Sampling temperature forwarded to the provider.
            function_lemmas: Extra per-call exempt lemmas (e.g. verb surface forms).
            cefr_lookup: ``{lemma: cefr_level}`` mapping from the imported lexicon.
                Required for CEFR-aware tolerance; without it all extras are violations.

        Raises:
            RuntimeError: If violations persist after ``max_retries``.
        """
        min_words, max_words = _target_length(len(new_words), cefr)
        messages = self._build_initial_messages(
            lesson_id,
            language,
            cefr,
            theme,
            new_words,
            min_words,
            max_words,
            outline=outline,
            communicative_goals=communicative_goals,
            glosses=glosses,
            verb_lemmas=verb_lemmas,
            english_seed_words=english_seed_words,
        )
        raw_messages = [m.as_dict() for m in messages]
        resolved_model = model or ""
        diagnostics: list[AttemptDiagnostics] = []
        best_content: str | None = None
        best_result: ValidationResult | None = None
        best_attempt: int | None = None

        for attempt in range(1, self._max_retries + 1):
            # Cache: only the first attempt (deterministic prompt) is cached.
            if attempt == 1 and self._cache is not None:
                cached = self._cache.get(resolved_model, raw_messages)
                if cached is not None:
                    self._log_messages(attempt, messages)
                    parsed_title, parsed_text = _extract_title_and_text(cached.content)
                    result: ValidationResult = self._validator.validate(
                        parsed_text,
                        allowed_words,
                        extra_function_lemmas=function_lemmas,
                        cefr_target=cefr,
                        cefr_lookup=cefr_lookup,
                        extra_tolerance=self._extra_tolerance,
                        new_word_count=len(new_words),
                    )
                    return GeneratedLesson(
                        lesson_id=lesson_id,
                        content=parsed_text,
                        attempts=1,
                        title=parsed_title,
                        tolerated=result.tolerated,
                        violations=result.violations,
                        fallback=not result.is_valid,
                        best_attempt=1,
                        diagnostics=(
                            AttemptDiagnostics(
                                attempt=1,
                                violations=result.violations,
                                tolerated=result.tolerated,
                            ),
                        ),
                    )

            try:
                self._log_messages(attempt, messages)
                response = self._provider.complete(
                    messages, model=model, temperature=temperature
                )
            except LLMError:
                if not self._fail_open_on_llm_error:
                    raise
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=_fallback_content(
                        new_words, lesson_id=lesson_id, reason="llm-timeout"
                    ),
                    attempts=attempt,
                    fallback=True,
                    best_attempt=attempt,
                    diagnostics=tuple(diagnostics),
                )

            if self._cache is not None and attempt == 1:
                self._cache.put(resolved_model, raw_messages, response)

            parsed_title, parsed_text = _extract_title_and_text(response.content)

            result: ValidationResult = self._validator.validate(
                parsed_text,
                allowed_words,
                extra_function_lemmas=function_lemmas,
                cefr_target=cefr,
                cefr_lookup=cefr_lookup,
                extra_tolerance=self._extra_tolerance,
                new_word_count=len(new_words),
            )
            diagnostics.append(
                AttemptDiagnostics(
                    attempt=attempt,
                    violations=result.violations,
                    tolerated=result.tolerated,
                )
            )

            if (
                best_result is None
                or len(result.violations) < len(best_result.violations)
                or (
                    len(result.violations) == len(best_result.violations)
                    and parsed_text.strip()
                    and best_content is not None
                    and len(parsed_text) > len(best_content)
                )
            ):
                best_content = parsed_text
                best_result = result
                best_attempt = attempt

            if result.is_valid:
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=parsed_text,
                    title=parsed_title,
                    attempts=attempt,
                    tolerated=result.tolerated,
                    best_attempt=attempt,
                    diagnostics=tuple(diagnostics),
                )

            if self._retry_strategy == "corrective":
                # Multi-turn corrective refinement: ask the model to revise the
                # previous draft by explicitly removing current violations.
                messages = [
                    *messages,
                    Message(Role.ASSISTANT, response.content),
                    Message(Role.USER, _feedback_message(result, cefr)),
                ]

        if self._fail_open_on_validation_error:
            if (
                best_content is not None
                and best_result is not None
                and best_attempt is not None
            ):
                return GeneratedLesson(
                    lesson_id=lesson_id,
                    content=best_content,
                    attempts=self._max_retries,
                    tolerated=best_result.tolerated,
                    fallback=True,
                    violations=best_result.violations,
                    best_attempt=best_attempt,
                    diagnostics=tuple(diagnostics),
                )

            return GeneratedLesson(
                lesson_id=lesson_id,
                content=_fallback_content(
                    new_words, lesson_id=lesson_id, reason="validation-retries"
                ),
                attempts=self._max_retries,
                fallback=True,
                best_attempt=self._max_retries,
                diagnostics=tuple(diagnostics),
            )

        raise RuntimeError(
            f"LessonGenerator exceeded max_retries={self._max_retries} for lesson {lesson_id!r}. "
            "Vocabulary violations could not be resolved."
        )
