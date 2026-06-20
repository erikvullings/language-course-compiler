# 0014 Minimal-edit refinement loop

Status: done
Priority: medium
Owner: Erik Vullings
Agent: claude
Area: generation
Depends on: none

## Context

On a vocabulary violation `LessonGenerator` already re-sends the whole
conversation, including the previous (bad) story as an assistant turn
(`lesson.py`). But `_feedback_message` says *"Please rewrite the lesson"*, which
invites a fresh from-scratch draft. Fresh resampling is noisy: each retry can
swap one set of offending words for a different set, so the loop descends
erratically and may oscillate.

Instruct **minimal revision** instead: keep the previous version, change only the
flagged words, preserve title/structure and roughly the same length. This turns
each retry from a fresh stochastic sample into a local edit that descends
monotonically toward a valid lesson.

## Acceptance Criteria

- `_feedback_message` instructs the model to revise its previous version with the
  smallest possible change (explicitly: do not rewrite from scratch; keep
  structure and length), while still naming the words to remove/replace.
- The offending words remain listed in the feedback (existing behaviour kept).
- Multi-turn conversation structure is unchanged (assistant + user feedback
  appended).
- Tests assert the minimal-edit framing and that violations are still named.
  Suite green, ruff clean.

## Implementation Notes

- `src/course_compiler/generation/lesson.py`: `_feedback_message`.
- Caching caveat (not in scope here): only attempt 1 is cached, so a
  multi-turn-refined result is not yet reproducible. Track separately if needed.

## Agent Notes

- Initial note: third of the three agreed changes. Small, prompt-only.
- Done 2026-06-20. `lesson.py` `_feedback_message` now instructs a minimal
  revision of the previous draft (keep title/structure/length, "Do not rewrite
  from scratch") and to remove/replace only the named words. Test
  `test_retry_feedback_asks_for_minimal_edit_not_rewrite` in
  `test_lesson_generator.py`. 150 passed, lesson.py ruff clean.
