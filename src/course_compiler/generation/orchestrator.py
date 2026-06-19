"""LessonOrchestrator: filter → theme → sequence → generate.

Turns a flat list of imported :class:`~course_compiler.models.Word` objects into
a sequence of generated lessons for a target CEFR level.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from course_compiler.generation.lesson import GeneratedLesson, LessonGenerator
from course_compiler.generation.themes import ThemeAssigner
from course_compiler.models import PartOfSpeech, Word

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


@dataclass(frozen=True)
class LessonPlan:
    lesson_id: str
    theme: str
    new_words: list[Word]
    allowed_lemmas: set[str]
    function_lemmas: set[str]


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

    def plan(self, words: list[Word], *, cefr: str) -> list[LessonPlan]:
        """Build an ordered list of :class:`LessonPlan` without calling the LLM generator.

        Steps:
        1. Filter to the target CEFR level (words with no CEFR tag are excluded).
        2. Split into function words (always-allowed) and content words (validated).
        3. Sort content words by frequency rank.
        4. Ask the theme assigner to cluster content words.
        5. Slice each theme into lessons of ``words_per_lesson`` words.
        6. Accumulate ``allowed_lemmas`` across lessons.
        """
        cefr_words = [w for w in words if w.cefr == cefr]
        function_lemmas = {w.lemma for w in cefr_words if self._is_function(w)}
        content_words = sorted(
            [w for w in cefr_words if not self._is_function(w)],
            key=self._sort_key,
        )

        if not content_words:
            return []

        themes = self._assigner.assign(content_words)
        # Deterministic theme order: alphabetical by theme name.
        ordered_themes = sorted(themes.items())

        plans: list[LessonPlan] = []
        accumulated: set[str] = set()
        lesson_num = 1

        for theme_name, theme_words in ordered_themes:
            sorted_theme = sorted(theme_words, key=self._sort_key)
            for i in range(0, len(sorted_theme), self._words_per_lesson):
                batch = sorted_theme[i : i + self._words_per_lesson]
                new_lemmas = {w.lemma for w in batch}
                plans.append(
                    LessonPlan(
                        lesson_id=f"lesson{lesson_num:03d}",
                        theme=theme_name,
                        new_words=batch,
                        allowed_lemmas=accumulated | new_lemmas,
                        function_lemmas=function_lemmas,
                    )
                )
                accumulated |= new_lemmas
                lesson_num += 1

        return plans

    def generate(
        self,
        words: list[Word],
        *,
        language: str,
        cefr: str,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> list[GeneratedLesson]:
        """Plan then generate all lessons, returning :class:`GeneratedLesson` objects."""
        # Build a full CEFR lookup from the entire word list (not just the target level)
        # so the validator can classify extra words the LLM might introduce.
        cefr_lookup: dict[str, str] = {w.lemma: w.cefr for w in words if w.cefr is not None}

        plans = self.plan(words, cefr=cefr)
        lessons: list[GeneratedLesson] = []
        for plan in plans:
            lesson = self._generator.generate(
                plan.lesson_id,
                [w.lemma for w in plan.new_words],
                plan.allowed_lemmas,
                language=language,
                cefr=cefr,
                theme=plan.theme,
                model=model,
                temperature=temperature,
                function_lemmas=plan.function_lemmas,
                cefr_lookup=cefr_lookup,
            )
            lessons.append(lesson)
        return lessons
