# 0012 Decouple lesson text length from new-word count

Status: done
Priority: high
Owner: Erik Vullings
Agent: claude
Area: generation
Depends on: none

## Context

`LessonGenerator` derives the requested text length from the number of *new*
words only: `_target_length(len(new_words))` = `15 * new_words` (`lesson.py`).

Early in a course the allowed vocabulary equals the new words (nothing has
accumulated yet), so lesson 1 with 10 new words demands ~150 words of natural
prose built from a 10-word vocabulary. That is not hard, it is impossible to do
*naturally* — hence stilted, repetitive output. This is the root cause of the
"15 words from nothing" cold-start problem.

The fix: length must reflect the **recombinant vocabulary the learner actually
has** (the allowed set), not how many words are new this lesson. Concretely,
length is the *minimum* of two budgets:

- a per-new-word context budget (`WORDS_PER_NEW_WORD * new_count`), and
- a vocabulary-sustainability budget (`WORDS_PER_ALLOWED_WORD * allowed_count`).

Early lessons become vocabulary-limited (short, natural); later lessons, where
the allowed set dwarfs the new words, stay new-word-limited (the current
behaviour). One small change handles both regimes.

## Acceptance Criteria

- `_target_length` takes both the new-word count and the allowed-vocab count and
  returns `min(by_new, by_vocab)`, floored at a sensible minimum.
- For a cold-start lesson (allowed ≈ new), the requested length is small
  (vocabulary-limited), not `15 × new_words`.
- For a mature lesson (allowed ≫ new), the requested length is unchanged
  (`15 × new_words`).
- `LessonGenerator.generate` passes `len(allowed_words)` into the length
  computation.
- Existing length tests updated; new cold-start test added. Full suite green,
  ruff clean.

## Implementation Notes

- `src/course_compiler/generation/lesson.py`: `_target_length`, `generate`.
- Add `WORDS_PER_ALLOWED_WORD` constant (~4: roughly how many words of
  low-repetition text N content words sustain).
- `allowed_words` already excludes function words / verb surface forms — content
  vocabulary is the right driver of sustainability.

## Agent Notes

- Initial note: first of the three lesson-content-generation changes agreed with
  Erik. Smallest change, highest leverage — unblocks the cold-start problem.
- Done 2026-06-20. `lesson.py`: `_target_length(new, allowed)` returns
  `max(min(15*new, 4*allowed), 30)`; added `WORDS_PER_ALLOWED_WORD=4`,
  `MIN_TARGET_WORDS=30`. `generate` passes `len(allowed_words)`. Tests in
  `test_lesson_generator.py` updated (new-word-limited, cold-start vocab-limited,
  floor). 145 passed, ruff clean.
