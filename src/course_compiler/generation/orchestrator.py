"""LessonOrchestrator: filter → theme → sequence → generate.

Turns a flat list of imported :class:`~course_compiler.models.Word` objects into
a sequence of generated lessons for a target CEFR level.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from course_compiler.generation.lesson import GeneratedLesson, LessonGenerator
from course_compiler.generation.themes import LessonThemePlan, ThemeAssigner
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

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

# Lightweight Dutch stopword-like lemmas to avoid as seed vocabulary where possible.
_LOW_SIGNAL_LEMMAS: frozenset[str] = frozenset(
    {
        "op",
        "te",
        "niet",
        "er",
        "maar",
        "ook",
        "uit",
        "dan",
        "was",
        "over",
        "nog",
        "zo",
        "wel",
        "nu",
        "dus",
        "hier",
        "weer",
        "alleen",
        "onder",
        "tussen",
        "bij",
        "door",
        "naar",
        "om",
    }
)


def _lesson_sort_key(lesson_id: str) -> tuple[int, str]:
    """Sort keys like lesson001, lesson002, ... in numeric order."""
    m = re.search(r"(\d+)$", lesson_id)
    if m is None:
        return (999_999, lesson_id)
    return (int(m.group(1)), lesson_id)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 1}


def _gloss_primary_term(gloss: str) -> str:
    """The primary English term of a translation gloss.

    Glosses look like ``"house"``, ``"street (a paved road)"`` or
    ``"day (period of 24 hours)"``; the leading term before any parenthesis,
    comma or semicolon is the headword we match seed words against.
    """
    head = gloss.split("(")[0]
    for sep in (",", ";"):
        head = head.split(sep)[0]
    return head.strip().lower()


def _resolve_seed_words(words: list[Word], seed_words: list[str]) -> list[str]:
    """Map English ``seed_words`` to lexicon lemmas via their English glosses.

    Deterministic and independent of any LLM: for each seed, the most frequent
    word whose primary English gloss equals the seed is chosen (one lemma per
    seed). Lets a catalog anchor lessons in concrete vocabulary that the
    frequency fallback would otherwise bury under function words.
    """
    if not seed_words:
        return []

    index: dict[str, list[Word]] = {}
    for word in words:
        gloss = word.translations.get("en", "")
        term = _gloss_primary_term(gloss) if gloss else ""
        if term:
            index.setdefault(term, []).append(word)

    def rank(word: Word) -> int:
        if word.frequency and word.frequency.rank is not None:
            return word.frequency.rank
        return 999_999

    resolved: list[str] = []
    seen: set[str] = set()
    for seed in seed_words:
        key = seed.strip().lower()
        for word in sorted(index.get(key, []), key=rank):
            if word.lemma not in seen:
                seen.add(word.lemma)
                resolved.append(word.lemma)
                break
    return resolved


def _theme_candidate_pool(
    *,
    remaining_words: list[Word],
    theme: str,
    communicative_goals: list[str],
    cap: int = 200,
) -> list[str]:
    """Rank CEFR lemmas for a lesson theme and return a manageable candidate pool."""
    if not remaining_words:
        return []

    theme_tokens = _tokens(theme)
    for goal in communicative_goals:
        theme_tokens |= _tokens(goal)

    scored: list[tuple[int, int, str]] = []
    for word in remaining_words:
        score = 0

        # Prefer nouns/verbs/adjectives/interjections as lesson seed lemmas.
        if word.part_of_speech in {
            PartOfSpeech.NOUN,
            PartOfSpeech.VERB,
            PartOfSpeech.ADJECTIVE,
            PartOfSpeech.INTERJECTION,
            PartOfSpeech.NUMERAL,
        }:
            score += 2

        # Penalize low-signal lemmas that often reduce theme coherence.
        if word.lemma.lower() in _LOW_SIGNAL_LEMMAS:
            score -= 5

        lexical_tokens: set[str] = _tokens(word.lemma)
        lexical_tokens |= _tokens(" ".join(word.tags))
        lexical_tokens |= _tokens(" ".join(word.related))
        lexical_tokens |= _tokens(" ".join(word.synonyms))
        lexical_tokens |= _tokens(" ".join(word.antonyms))
        lexical_tokens |= _tokens(" ".join(word.translations.values()))

        overlap = len(theme_tokens & lexical_tokens)
        if overlap > 0:
            score += overlap * 6

        # More frequent words first when scores are tied.
        rank = (
            word.frequency.rank
            if word.frequency and word.frequency.rank is not None
            else 999_999
        )
        scored.append((score, -rank, word.lemma))

    scored.sort(reverse=True)
    candidate_lemmas = [lemma for _, _, lemma in scored]
    return candidate_lemmas[: max(1, cap)]


def _load_predefined_themes(path: Path) -> dict[str, list[LessonThemePlan]]:
    """Load {CEFR: [LessonThemePlan, ...]} from a YAML catalog file."""
    if not path.exists():
        return {}

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}

    result: dict[str, list[LessonThemePlan]] = {}
    for cefr_key, cefr_block in loaded.items():
        if not isinstance(cefr_key, str) or not isinstance(cefr_block, dict):
            continue

        entries: list[tuple[str, LessonThemePlan]] = []
        for lesson_id, lesson_data in cefr_block.items():
            if not isinstance(lesson_id, str):
                continue

            theme_name = ""
            communicative_goals: list[str] = []
            english_seed_words: list[str] = []
            outline = ""
            if isinstance(lesson_data, dict):
                raw_theme = lesson_data.get("theme")
                if isinstance(raw_theme, str):
                    theme_name = raw_theme.strip()
                raw_goals = lesson_data.get("communicativeGoals")
                if isinstance(raw_goals, list):
                    communicative_goals = [
                        str(goal).strip()
                        for goal in raw_goals
                        if isinstance(goal, str) and goal.strip()
                    ]
                raw_seeds = lesson_data.get("seedWords")
                if isinstance(raw_seeds, list):
                    english_seed_words = [
                        str(word).strip()
                        for word in raw_seeds
                        if isinstance(word, str) and word.strip()
                    ]
                raw_outline = lesson_data.get("outline")
                if isinstance(raw_outline, str):
                    outline = raw_outline.strip()
            elif isinstance(lesson_data, str):
                theme_name = lesson_data.strip()

            if theme_name:
                entries.append(
                    (
                        lesson_id,
                        LessonThemePlan(
                            theme=theme_name,
                            seed_lemmas=[],
                            communicative_goals=communicative_goals,
                            english_seed_words=english_seed_words,
                            outline=outline,
                        ),
                    )
                )

        if entries:
            entries.sort(key=lambda item: _lesson_sort_key(item[0]))
            result[cefr_key.upper()] = [plan for _, plan in entries]

    return result


def _verb_surface_forms(verb: Verb) -> set[str]:
    """Collect every conjugated surface form from all tense tables of a verb."""
    forms: set[str] = set()
    for table in (
        verb.present,
        verb.past,
        verb.perfect,
        verb.imperative,
        verb.future,
        verb.conditional,
        verb.subjunctive,
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


def _is_verb_item(word: Word, verb_lookup: dict[str, Verb]) -> bool:
    """True if *word* is the verb sense of a (lemma, pos) item, not a noun sense.

    The learnable unit is ``(lemma, part_of_speech)``: a noun and a verb sharing a
    surface form (e.g. ``eten`` = food / to eat) are distinct items. A verb stub
    carries ``part_of_speech == VERB``; a homographic noun keeps its own POS, so a
    lemma being present in ``verb_lookup`` no longer means *every* item with that
    lemma is a verb — only the VERB-tagged stub is.
    """
    return word.part_of_speech == PartOfSpeech.VERB and word.lemma in verb_lookup


def _group_by_lemma(words: list[Word]) -> dict[str, list[Word]]:
    """Map each lemma to all its sense-items (noun + verb homographs co-exist)."""
    grouped: dict[str, list[Word]] = {}
    for word in words:
        grouped.setdefault(word.lemma, []).append(word)
    return grouped


def _split_batch(
    batch: list[Word], verb_lookup: dict[str, Verb]
) -> tuple[list[Word], list[Verb], set[str]]:
    """Split a batch into (non-verb words, verbs, verb surface forms) by item POS."""
    batch_verbs = [verb_lookup[w.lemma] for w in batch if _is_verb_item(w, verb_lookup)]
    non_verb_batch = [w for w in batch if not _is_verb_item(w, verb_lookup)]
    new_forms: set[str] = set()
    for verb in batch_verbs:
        new_forms |= _verb_surface_forms(verb)
    return non_verb_batch, batch_verbs, new_forms


@dataclass(frozen=True)
class LessonPlan:
    lesson_id: str
    theme: str
    new_words: list[Word]
    allowed_lemmas: set[str]
    function_lemmas: set[str]
    new_verbs: list[Verb] = field(default_factory=list)
    allowed_forms: set[str] = field(default_factory=set)
    outline: str = ""


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
        first_lesson_words: int | None = None,
        front_load_lessons: int = 3,
        function_pos: frozenset[PartOfSpeech] = FUNCTION_POS,
        predefined_themes_path: Path | None = None,
        predefined_themes: dict[str, list[str] | list[LessonThemePlan]] | None = None,
    ) -> None:
        self._generator = generator
        self._assigner = theme_assigner
        self._words_per_lesson = words_per_lesson
        self._first_lesson_words = first_lesson_words
        self._front_load_lessons = front_load_lessons
        self._function_pos = function_pos
        if predefined_themes is not None:
            self._predefined_themes = {
                key.upper(): [
                    (
                        plan
                        if isinstance(plan, LessonThemePlan)
                        else LessonThemePlan(theme=str(plan), seed_lemmas=[])
                    )
                    for plan in value
                ]
                for key, value in predefined_themes.items()
            }
        elif predefined_themes_path is not None:
            self._predefined_themes = _load_predefined_themes(predefined_themes_path)
        else:
            self._predefined_themes = {}

    def _is_function(self, word: Word) -> bool:
        return word.part_of_speech in self._function_pos

    def _sort_key(self, word: Word) -> int:
        if word.frequency and word.frequency.rank is not None:
            return word.frequency.rank
        return 999_999

    def _budget_for(self, lesson_number: int) -> int:
        """New-word budget for lesson *lesson_number* (1-based).

        Uniform (``words_per_lesson``) unless ``first_lesson_words`` is set, in
        which case the budget tapers linearly from ``first_lesson_words`` (lesson
        1) down to ``words_per_lesson`` over ``front_load_lessons`` lessons, then
        holds at the steady state. This front-loading gives early lessons enough
        critical mass to form coherent text when there is no prior vocabulary to
        recombine (cf. the Delft Method).
        """
        steady = self._words_per_lesson
        first = self._first_lesson_words
        if first is None:
            return steady
        if lesson_number >= self._front_load_lessons:
            return steady
        span = self._front_load_lessons - 1
        if span <= 0:
            return max(1, first)
        frac = (lesson_number - 1) / span
        return max(1, round(first + frac * (steady - first)))

    def _distribute(self, total: int, lessons: int) -> list[tuple[int, int]]:
        """Split ``total`` ordered items into exactly ``lessons`` contiguous
        ``(start, end)`` ranges that together cover everything.

        Even split by default; front-loaded (early lessons larger) when
        ``first_lesson_words`` is configured. Used by the catalog path so every
        configured theme becomes one lesson and no vocabulary is dropped.
        """
        if lessons <= 0 or total <= 0:
            return []
        if lessons >= total:
            return [(i, i + 1) for i in range(total)]

        if self._first_lesson_words is None:
            bounds = [i * total // lessons for i in range(lessons + 1)]
        else:
            weights = [max(1, self._budget_for(i + 1)) for i in range(lessons)]
            wsum = sum(weights)
            bounds = [0]
            acc = 0
            for w in weights:
                acc += w
                bounds.append(round(acc / wsum * total))
            bounds[-1] = total

        return [(bounds[i], bounds[i + 1]) for i in range(lessons)]

    def _plan_from_blueprints(
        self,
        blueprints: list[LessonThemePlan],
        *,
        all_content: list[Word],
        function_lemmas: set[str],
        verb_lookup: dict[str, Verb],
    ) -> list[LessonPlan]:
        by_lemma = _group_by_lemma(all_content)
        accumulated: set[str] = set()
        accumulated_forms: set[str] = set()
        seen_new_lemmas: set[str] = set()
        plans: list[LessonPlan] = []
        lesson_num = 1

        for blueprint in blueprints:
            batch: list[Word] = []
            for lemma in blueprint.seed_lemmas:
                items = by_lemma.get(lemma)
                if not items or lemma in seen_new_lemmas:
                    continue
                # A seed lemma pulls in all its senses (noun + verb homograph).
                batch.extend(items)
                seen_new_lemmas.add(lemma)

            if not batch:
                continue

            new_lemmas = {w.lemma for w in batch}
            non_verb_batch, batch_verbs, new_forms = _split_batch(batch, verb_lookup)

            plans.append(
                LessonPlan(
                    lesson_id=f"lesson{lesson_num:03d}",
                    theme=blueprint.theme,
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

        # Ensure full coverage even if the LLM misses lemmas.
        leftover = [w for w in all_content if w.lemma not in seen_new_lemmas]
        for i in range(0, len(leftover), self._words_per_lesson):
            batch = leftover[i : i + self._words_per_lesson]
            new_lemmas = {w.lemma for w in batch}
            non_verb_batch, batch_verbs, new_forms = _split_batch(batch, verb_lookup)

            plans.append(
                LessonPlan(
                    lesson_id=f"lesson{lesson_num:03d}",
                    theme="misc",
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

    def _plan_with_theme_sequence(
        self,
        theme_sequence: list[LessonThemePlan],
        *,
        cefr: str,
        all_content: list[Word],
        function_lemmas: set[str],
        verb_lookup: dict[str, Verb],
        language: str = "",
    ) -> list[LessonPlan]:
        """Use predefined lesson theme names in order.

        When the configured theme sequence is longer than the default lesson
        count implied by ``words_per_lesson``, spread the content across the
        configured sequence so every configured lesson theme can be used.
        """
        plans: list[LessonPlan] = []
        accumulated: set[str] = set()
        accumulated_forms: set[str] = set()

        if not all_content:
            return plans

        # One lesson per configured theme: distribute ALL content across the
        # themes so nothing is dropped (front-loaded when configured). The lesson
        # count is min(themes, words); words_per_lesson does not cap this path.
        lesson_count = min(len(theme_sequence), len(all_content))
        slices = self._distribute(len(all_content), lesson_count)

        # A lemma can map to multiple sense-items (noun + verb homograph); keep
        # them all so neither sense is dropped when the lemma is selected.
        by_lemma = _group_by_lemma(all_content)
        by_lemma_lower: dict[str, list[Word]] = {}
        for lemma, items in by_lemma.items():
            by_lemma_lower.setdefault(lemma.lower(), []).extend(items)
        ordered_lemmas = [w.lemma for w in all_content]
        used_lemmas: set[str] = set()
        proposer = getattr(self._assigner, "propose_theme_vocabulary", None)
        selector = getattr(self._assigner, "select_seed_lemmas_for_theme", None)

        # Reserve each theme's resolvable seed words globally: a lemma is "owned" by
        # the first theme whose seeds resolve to it, so an earlier theme's frequency
        # fallback can't steal a concrete noun a later theme is anchored on (e.g.
        # 'Nature' grabbing 'hond' before 'Animals'). A theme may still take a lemma
        # owned by an *earlier* theme that didn't use it, so coverage is preserved.
        seed_owner: dict[str, int] = {}
        for theme_index, plan_for_owner in enumerate(theme_sequence):
            for lemma in _resolve_seed_words(
                all_content, plan_for_owner.english_seed_words
            ):
                seed_owner.setdefault(lemma, theme_index)

        for index, (start, end) in enumerate(slices):
            if index >= len(theme_sequence):
                break

            theme_plan = theme_sequence[index]
            target_count = max(1, end - start)
            selected: list[str] = []
            remaining_words = [w for w in all_content if w.lemma not in used_lemmas]
            candidate_lemmas = _theme_candidate_pool(
                remaining_words=remaining_words,
                theme=theme_plan.theme,
                communicative_goals=theme_plan.communicative_goals,
            )

            # Highest priority: concrete anchor words resolved from the catalog's
            # English seed words via the lexicon's English glosses. Deterministic and
            # independent of LLM quality, so early lessons get scene-grounding nouns
            # instead of the high-frequency function words the fallback would pick.
            for lemma in _resolve_seed_words(
                remaining_words, theme_plan.english_seed_words
            ):
                if lemma not in selected:
                    selected.append(lemma)
                if len(selected) >= target_count:
                    break

            # Next: the LLM proposes theme-relevant words from its own knowledge;
            # we keep only those present in our lexicon, ranked by frequency. This
            # gives communicatively coherent vocabulary instead of frequency-noise.
            if len(selected) < target_count and callable(proposer):
                proposer_kwargs = dict(
                    cefr=cefr,
                    theme=theme_plan.theme,
                    communicative_goals=theme_plan.communicative_goals,
                    target_count=target_count,
                    already_used=sorted(used_lemmas),
                    language=language,
                )
                try:
                    # English seed-word anchors bias selection toward concrete nouns.
                    proposed = proposer(
                        **proposer_kwargs,
                        seed_words=theme_plan.english_seed_words,
                    )
                except TypeError:
                    # Back-compat with assigners that predate the seed_words arg.
                    proposed = proposer(**proposer_kwargs)
                survivors: list[Word] = []
                seen_lower: set[str] = set()
                for raw in proposed if isinstance(proposed, list) else []:
                    if not isinstance(raw, str):
                        continue
                    key = raw.strip().lower()
                    if not key or key in seen_lower:
                        continue
                    items = by_lemma.get(raw.strip()) or by_lemma_lower.get(key)
                    if not items or items[0].lemma in used_lemmas:
                        continue
                    owner = seed_owner.get(items[0].lemma)
                    if owner is not None and owner > index:
                        continue  # reserved for a later theme's seed
                    seen_lower.add(key)
                    # Use the first (most frequent) sense as the frequency proxy;
                    # all senses are pulled in when the lemma is batched below.
                    survivors.append(items[0])
                survivors.sort(key=self._sort_key)
                for word in survivors:
                    if word.lemma not in selected:
                        selected.append(word.lemma)
                    if len(selected) >= target_count:
                        break

            if len(selected) < target_count and callable(selector):
                try:
                    proposed = selector(
                        cefr=cefr,
                        theme=theme_plan.theme,
                        communicative_goals=theme_plan.communicative_goals,
                        target_count=target_count,
                        already_used=sorted(used_lemmas),
                        candidate_lemmas=candidate_lemmas,
                    )
                except TypeError:
                    # Backward compatibility with assigners that have the old signature.
                    proposed = selector(
                        cefr=cefr,
                        theme=theme_plan.theme,
                        communicative_goals=theme_plan.communicative_goals,
                        target_count=target_count,
                        already_used=sorted(used_lemmas),
                    )
                if not isinstance(proposed, list):
                    proposed = []
                for lemma in proposed:
                    if (
                        lemma in by_lemma
                        and lemma not in used_lemmas
                        and lemma not in selected
                    ):
                        selected.append(lemma)
                    if len(selected) >= target_count:
                        break

            if len(selected) < target_count:
                for lemma in candidate_lemmas + ordered_lemmas:
                    if lemma in used_lemmas or lemma in selected:
                        continue
                    owner = seed_owner.get(lemma)
                    if owner is not None and owner > index:
                        continue  # reserved for a later theme's seed
                    selected.append(lemma)
                    if len(selected) >= target_count:
                        break

            # Expand each selected lemma to all its sense-items so a noun and its
            # verb homograph are both taught (neither sense is silently dropped).
            batch = [item for lemma in selected for item in by_lemma.get(lemma, [])]
            if not batch:
                continue

            new_lemmas = {w.lemma for w in batch}
            non_verb_batch, batch_verbs, new_forms = _split_batch(batch, verb_lookup)

            theme_name = theme_plan.theme.strip() or "misc"
            plans.append(
                LessonPlan(
                    lesson_id=f"lesson{index + 1:03d}",
                    theme=theme_name,
                    new_words=non_verb_batch,
                    allowed_lemmas=accumulated | new_lemmas,
                    function_lemmas=function_lemmas,
                    new_verbs=batch_verbs,
                    allowed_forms=accumulated_forms | new_forms,
                    outline=theme_plan.outline,
                )
            )
            accumulated |= new_lemmas
            accumulated_forms |= new_forms
            used_lemmas |= new_lemmas

        return plans

    def plan(
        self,
        words: list[Word],
        *,
        cefr: str,
        verbs: list[Verb] | None = None,
        language: str = "",
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
        for verb in verbs or []:
            if verb.cefr == cefr:
                stub = _verb_as_word(verb)
                verb_stubs.append(stub)
                verb_lookup[verb.infinitive] = verb

        all_content = sorted(content_words + verb_stubs, key=self._sort_key)

        if not all_content:
            return []

        predefined_themes = self._predefined_themes.get(cefr.upper(), [])
        if predefined_themes:
            return self._plan_with_theme_sequence(
                predefined_themes,
                cefr=cefr,
                all_content=all_content,
                function_lemmas=function_lemmas,
                verb_lookup=verb_lookup,
                language=language,
            )

        planner = getattr(self._assigner, "plan_lessons", None)
        if callable(planner):
            planned_result = planner(
                all_content,
                cefr=cefr,
                words_per_lesson=self._words_per_lesson,
            )
            lesson_blueprints: list[LessonThemePlan] = []
            if isinstance(planned_result, list) and all(
                isinstance(item, LessonThemePlan) for item in planned_result
            ):
                lesson_blueprints = planned_result

            if lesson_blueprints:
                planned = self._plan_from_blueprints(
                    lesson_blueprints,
                    all_content=all_content,
                    function_lemmas=function_lemmas,
                    verb_lookup=verb_lookup,
                )
                if planned:
                    return planned

        themes = self._assigner.assign(all_content)
        # Deterministic theme order: alphabetical by theme name.
        ordered_themes = sorted(themes.items())

        plans: list[LessonPlan] = []
        accumulated: set[str] = set()
        accumulated_forms: set[str] = set()
        lesson_num = 1

        for theme_name, theme_words in ordered_themes:
            sorted_theme = sorted(theme_words, key=self._sort_key)
            # Front-loaded budget: the per-lesson size follows the global lesson
            # counter, so early lessons across the whole course get more words.
            cursor = 0
            while cursor < len(sorted_theme):
                budget = self._budget_for(lesson_num)
                batch = sorted_theme[cursor : cursor + budget]
                cursor += len(batch)
                new_lemmas = {w.lemma for w in batch}

                # Resolve verb stubs back to Verb objects; collect their surface forms.
                non_verb_batch, batch_verbs, new_forms = _split_batch(
                    batch, verb_lookup
                )

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

    def generate_iter(
        self,
        words: list[Word],
        *,
        language: str,
        cefr: str,
        verbs: list[Verb] | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> Iterator[tuple[LessonPlan, GeneratedLesson]]:
        """Plan, then generate lessons one at a time, yielding ``(plan, lesson)``.

        Streaming so callers can persist and report each lesson as it is produced
        instead of waiting for the whole (potentially long) batch to finish.
        """
        # Build a full CEFR lookup from words and verbs so the validator can classify
        # extra words the LLM might introduce.
        cefr_lookup: dict[str, str] = {
            w.lemma: w.cefr for w in words if w.cefr is not None
        }
        for verb in verbs or []:
            if verb.cefr is not None:
                cefr_lookup[verb.infinitive] = verb.cefr

        plans = self.plan(words, cefr=cefr, verbs=verbs, language=language)
        for plan in plans:
            new_word_lemmas = [w.lemma for w in plan.new_words] + [
                v.infinitive for v in plan.new_verbs
            ]
            lesson = self._generator.generate(
                plan.lesson_id,
                new_word_lemmas,
                plan.allowed_lemmas,
                language=language,
                cefr=cefr,
                theme=plan.theme,
                outline=plan.outline,
                model=model,
                temperature=temperature,
                function_lemmas=plan.function_lemmas | plan.allowed_forms,
                cefr_lookup=cefr_lookup,
            )
            yield plan, lesson

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
        return [
            lesson
            for _plan, lesson in self.generate_iter(
                words,
                language=language,
                cefr=cefr,
                verbs=verbs,
                model=model,
                temperature=temperature,
            )
        ]
