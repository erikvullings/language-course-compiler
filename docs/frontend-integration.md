# Frontend integration: POS-tagged, sense-linked lesson tokens

This document is the contract for the `zin` frontend after the compiler's
`course annotate` pass (task 0023). It lets the frontend **delete its translation
workaround layer** and render lessons directly from compiler output.

## TL;DR

Each annotated lesson JSON now carries two new fields:

- **`tokens`** — the lesson text already tokenized and linked. Render it directly;
  do **not** re-tokenize the markdown.
- **`vocabulary`** — the lesson's new words, each with resolved POS, sense gloss,
  gender/article and a `ref`.

The lexicon (`words.json`) is now keyed by **`lemma|pos`** (homographs survive),
each entry has a clean **`glosses`** list, and there is a new
**`separable-verbs.json`**.

## What changed in the data

### `words.json` — keyed by `(lemma, POS)`

Word `id` is now composite: `"morgen|noun"`, `"morgen|adverb"`, `"lopen|noun"`, …
So both senses of a homograph now exist (previously the second POS was dropped,
which is why `morgen` had to be hardcoded as "morning / tomorrow"). New field
`glosses` is the cleaned candidate sense list; `translations.en` is unchanged
(the joined display string).

```jsonc
{ "id": "morgen|noun",    "lemma": "morgen", "partOfSpeech": "noun",
  "glosses": ["morning"],  "gender": "m" }
{ "id": "morgen|adverb",  "lemma": "morgen", "partOfSpeech": "adverb",
  "glosses": ["tomorrow"] }
```

Build your `WORDS` dictionary keyed by **`id`** (`lemma|pos`), not by lemma.
`audio.json` is still keyed by bare lemma (pronunciation is POS-independent).

### `verbs.json` — unchanged keying + new flags

Still keyed by infinitive. New fields: `glosses`, and for separable verbs
`separable: true` + `prefix` (e.g. `"voor"`), plus `reflexive: true` where known.

### `separable-verbs.json` — new file

```json
{ "voorstellen": { "prefix": "voor", "stem": "stellen" } }
```

Optional for the frontend (the compiler already fuses separable verbs in
`tokens`), but useful if you want the mapping client-side.

### Lesson JSON — `vocabulary[]` and `tokens[]`

```jsonc
{
  "id": "lesson027",
  "title": "...",
  "text": "De vrouw stelt zich voor. Tot morgen!",   // unchanged source
  "newWords": ["...", "..."],                          // unchanged (still emitted)

  "vocabulary": [
    { "lemma": "vrouw", "pos": "noun", "ref": "vrouw|noun",
      "gloss": "woman", "gender": "f", "article": "de" },
    { "lemma": "voorstellen", "pos": "verb", "ref": "voorstellen",
      "gloss": "introduce" }
  ],

  "tokens": [
    "De ",
    { "w": "vrouw", "ref": "vrouw|noun", "pos": "noun", "gloss": "woman" },
    " ",
    { "w": "stelt", "ref": "voorstellen", "pos": "verb",
      "span": ["stelt", "voor"] },
    " zich ",
    { "w": "voor", "ref": "voorstellen", "pos": "verb",
      "span": ["stelt", "voor"] },
    ". Tot ",
    { "w": "morgen", "ref": "morgen|adverb", "pos": "adverb", "gloss": "tomorrow" },
    "!"
  ]
}
```

## `tokens[]` rendering rule

`tokens` is an ordered stream of `string | object`:

- **string** → literal text (whitespace/punctuation/unlinked words). Render as-is.
- **object** → a linkable word. Shape:
  - `w` — surface form to display.
  - `ref` — key into `WORDS` (`lemma|pos`) **or** `VERBS` (infinitive). If `ref`
    contains `"|"` it's a word; otherwise it's a verb infinitive. (Or just look it
    up in `VERBS` first, then `WORDS`.)
  - `pos` — resolved part of speech (`"noun"`, `"verb"`, `"adverb"`, …).
  - `gloss` — the **correct sense for this occurrence**. Prefer this for the
    tooltip; fall back to `WORDS[ref].glosses[0]` / `translations.en` only if absent.
  - `span` — present only for separable verbs: the surface pieces of the one
    lexical unit (e.g. `["stelt", "voor"]`). Both the base and the particle tokens
    carry the same `ref` and `span`; show a single tooltip for the pair and
    optionally highlight them together.

Concatenating `string` items and the `w` of object items reproduces the original
text exactly, so you can render the stream in order without touching `text`.

Pseudocode:

```ts
for (const tok of lesson.tokens) {
  if (typeof tok === "string") { appendText(tok); continue; }
  const entry = VERBS[tok.ref] ?? WORDS[tok.ref];   // for card/audio/conjugation
  appendChip({ surface: tok.w, gloss: tok.gloss, pos: tok.pos, entry, span: tok.span });
}
```

## Workarounds to delete (in `scripts/generate-data.ts`)

All of these existed only because the compiler didn't emit POS/sense. They are now
redundant:

| Remove | Replaced by |
|---|---|
| `tokenizeLine` / `resolveRef` | `lesson.tokens` (already tokenized + linked) |
| `preferVerb` | per-token `pos`/`ref` (spaCy + verb-form map) |
| `isLikelyProperName` | proper names are already left unlinked (plain strings) |
| `WORD_EN_OVERRIDES` | `(lemma, pos)` entries + per-token `gloss` (e.g. `morgen`) |
| `VERB_EN_ALTERNATES` | verb `glosses` order + per-token `gloss` |
| `isWeakLexTranslation` | `glosses` already excludes "inflection of"/usage notes |
| `guessArticle` (suffix heuristic) | `gender` is present per noun; use it directly |
| `genderToArticle` | keep as a 1-liner, **or** use `vocabulary[].article` |

The previously-unsolvable separable-verb case (`stelt … voor` → `voorstellen`) is
handled: both tokens link to `voorstellen` and share `span`.

## Backward compatibility

A lesson that hasn't been annotated yet has **no** `tokens`/`vocabulary`. Guard:
if `lesson.tokens?.length`, render from it; otherwise keep the old tokenizer path
during migration.

## Producing the data (compiler side)

```bash
python -m spacy download nl_core_news_lg       # one-time, for POS tagging
course import --kaikki … --out courses/nl     # composite ids + glosses + separable-verbs.json
course annotate --lang nl --cefr A1            # writes tokens[]/vocabulary[] into each lesson
course annotate --lang nl --cefr A1 --only lesson003   # re-tag one edited lesson
```

`course annotate` is idempotent and runs on existing lessons without regenerating
them. After a manual edit to a lesson's `text`, re-run with `--only <id>`.

## Residual corrections (rare)

If a specific token still resolves wrong, the compiler reads an optional
`courses/<lang>/lessons/<CEFR>/<lessonId>.meta.yaml` sidecar:

```yaml
linkAs:                       # force a surface form to a ref ("" to unlink)
  groet: groeten
glossOverrides:               # force the display gloss for a ref or lemma
  "op|preposition": "on"
separableVerbs:               # force a fusion the detector missed
  - surface: "stelt voor"
    lemma: voorstellen
```

Re-run `course annotate --only <id>` after editing. These corrections live with the
content, not in the frontend.
