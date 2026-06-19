"""LessonOrchestrator: filter → theme → sequence → generate.

Turns a flat list of imported :class:`~course_compiler.models.Word` objects into
a sequence of generated lessons for a target CEFR level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from course_compiler.generation.lesson import GeneratedLesson, LessonGenerator
from course_compiler.generation.themes import ThemeAssigner
from course_compiler.models import PartOfSpeech, Verb, Word

# POS tags treated as function words — exempt from vocabulary validation and
# never listed as "new words" in a lesson.
FUNCTION_POS: frozenset[PartOfSpeech] = frozenset(
    {
        PartOfSpeech.ARTICLE,
        PartOfSpeech.CONJUNCTION,
        PartOfSpeech.PREPOSITION,
        PartOfSpeech.PRONOUN,
        PartOfSpeech.DETERMINER,
    }
)


def _verb_surface_forms(verb: Verb) -> set[str]:
    """Collect every conjugated surface form from all tense tables of a verb."""
    forms: set[str] = set()
    for table in (
        verb.present, verb.past, verb.perfect, verb.imperative,
        verb.future, verb.conditional, verb.subjunctive,
    ):
        forms.update(table.values())
    return forms


def _verb_as_word(verb: Verb) -> Word:
    """Create a minimal Word stub so a Verb participates in theme assignment."""
    return Word(
        id=verb.id,
        language=verb.language,
        lemma=verb.infinitive,
        normalized=verb.infinitive,
        part_of_speech=PartOfSpeech.VERB,
        frequency=verb.frequency,
        cefr=verb.cefr,
    )


@dataclass(frozen=True)
class LessonPlan:
    lesson_id: str
    theme: str
    new_words: list[Word]
    allowed_lemmas: set[str]
    function_lemmas: set[str]
    new_verbs: list[Verb] = field(default_factory=list)
    allowed_forms: set[str] = field(default_factory=set)


class LessonOrchestrator:
    """Plan and generate a sequence of lessons from an imported lexicon.

    Args:
        generator: A configured :class:`~course_compiler.generation.lesson.LessonGenerator`.
        theme_assigner: Strategy for grouping content words into semantic themes.
        words_per_lesson: Maximum new content words introduced per lesson.
        function_pos: POS tags treated as function words (exempt from validation).
    """

    def __init__(
        self,
        generator: LessonGenerator,
        theme_assigner: ThemeAssigner,
        *,
        words_per_lesson: int = 10,
        function_pos: frozenset[PartOfSpeech] = FUNCTION_POS,
    ) -> None:
        self._generator = generator
        self._assigner = theme_assigner
        self._words_per_lesson = words_per_lesson
        self._function_pos = function_pos

    def _is_function(self, word: Word) -> bool:
        return word.part_of_speech in self._function_pos

    def _sort_key(self, word: Word) -> int:
        if word.frequency and word.frequency.rank is not None:
            return word.frequency.rank
        return 999_999

    def plan(
        self,
        words: list[Word],
        *,
        cefr: str,
        verbs: list[Verb] | None = None,
    ) -> list[LessonPlan]:
        """Build an ordered list of :class:`LessonPlan` without calling the LLM generator.

        Steps:
        1. Filter to the target CEFR level (words/verbs with no CEFR tag are excluded).
        2. Split words into function words (always-allowed) and content words (validated).
        3. Verbs are treated as content words; their infinitive is the lesson-intro lemma.
        4. Sort content words + verb stubs by frequency rank.
        5. Ask the theme assigner to cluster content words (including verb stubs).
        6. Slice each theme into lessons of ``words_per_lesson`` words.
        7. Accumulate ``allowed_lemmas`` and ``allowed_forms`` (verb surface forms) across lessons.
        """
        cefr_words = [w for w in words if w.cefr == cefr]
        function_lemmas = {w.lemma for w in cefr_words if self._is_function(w)}
        content_words = [w for w in cefr_words if not self._is_function(w)]

        # Build verb lookup and create Word stubs so verbs participate in theme assignment.
        verb_lookup: dict[str, Verb] = {}
        verb_stubs: list[Word] = []
        for verb in (verbs or []):
            if verb.cefr == cefr:
                stub = _verb_as_word(verb)
                verb_stubs.append(stub)
                verb_lookup[verb.infinitive] = verb

        all_content = sorted(content_words + verb_stubs, key=self._sort_key)

        if not all_content:
            return []

        themes = self._assigner.assign(all_content)
        # Deterministic theme order: alphabetical by theme name.
        ordered_themes = sorted(themes.items())

        plans: list[LessonPlan] = []
        accumulated: set[str] = set()
        accumulated_forms: set[str] = set()
        lesson_num = 1

        for theme_name, theme_words in ordered_themes:
            sorted_theme = sorted(theme_words, key=self._sort_key)
            for i in range(0, len(sorted_theme), self._words_per_lesson):
                batch = sorted_theme[i : i + self._words_per_lesson]
                new_lemmas = {w.lemma for w in batch}

                # Resolve verb stubs back to Verb objects; collect their surface forms.
                batch_verbs = [verb_lookup[w.lemma] for w in batch if w.lemma in verb_lookup]
                non_verb_batch = [w for w in batch if w.lemma not in verb_lookup]
                new_forms: set[str] = set()
                for verb in batch_verbs:
                    new_forms |= _verb_surface_forms(verb)

                plans.append(
                    LessonPlan(
                        lesson_id=f"lesson{lesson_num:03d}",
                        theme=theme_name,
                        new_words=non_verb_batch,
                        allowed_lemmas=accumulated | new_lemmas,
                        function_lemmas=function_lemmas,
                        new_verbs=batch_verbs,
                        allowed_forms=accumulated_forms | new_forms,
                    )
                )
                accumulated |= new_lemmas
                accumulated_forms |= new_forms
                lesson_num += 1

        return plans

    def generate(
        self,
        words: list[Word],
        *,
        language: str,
        cefr: str,
        verbs: list[Verb] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> list[GeneratedLesson]:
        """Plan then generate all lessons, returning :class:`GeneratedLesson` objects."""
        # Build a full CEFR lookup from words and verbs so the validator can classify
        # extra words the LLM might introduce.
        cefr_lookup: dict[str, str] = {w.lemma: w.cefr for w in words if w.cefr is not None}
        for verb in (verbs or []):
            if verb.cefr is not None:
                cefr_lookup[verb.infinitive] = verb.cefr

        plans = self.plan(words, cefr=cefr, verbs=verbs)
        lessons: list[GeneratedLesson] = []
        for plan in plans:
            new_word_lemmas = [w.lemma for w in plan.new_words] + [v.infinitive for v in plan.new_verbs]
            lesson = self._generator.generate(
                plan.lesson_id,
                new_word_lemmas,
                plan.allowed_lemmas,
                language=language,
                cefr=cefr,
                theme=plan.theme,
                model=model,
                temperature=temperature,
                function_lemmas=plan.function_lemmas | plan.allowed_forms,
                cefr_lookup=cefr_lookup,
            )
            lessons.append(lesson)
        return lessons
