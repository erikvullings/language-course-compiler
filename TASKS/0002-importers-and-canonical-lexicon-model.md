# 0002 Importers and canonical lexicon model

Status: in_progress
Priority: high
Owner: erik.vullings
Agent: claude
Area: pipeline
Depends on: 0001

## Context
Define the internal data model (word, verb, grammar, lesson, example, exercise)
and importer plugins that normalize external sources into it. The compiler must
contain no language-specific logic.

## Acceptance Criteria
- [x] Canonical models matching the YAML schemas in `INITIAL_INSTRUCTIONS.md`
      (Pydantic, language-agnostic) — `course_compiler/models.py`
- [x] A Dutch importer mapping the provided sources to canonical YAML
- [ ] Generic pluggable importer interface + CSV/TSV/YAML importers (deferred)
- [x] One model per lemma, indexed by id; verbs vs words separated
- [x] Reproducible: deterministic YAML (camelCase, no sort, exclude None)

## Implementation Notes

- Importers convert their source into the common internal model (one plugin per
  format: YAML, JSON, CSV, TSV, XML, Wiktionary, OpenTaal, WordNet, SUBTLEX, ...).
- Keep language config + datasets external (e.g. `courses/<lang>/`).

To build your exact YAML structures completely within a standard open-source framework (e.g., MIT, Apache 2.0, or standard CC-BY), combine the following three core datasets:

### The Morphological & Core Lexicon: OpenTaal

OpenTaal is the backbone of Dutch open-source spellcheckers and morphological software. It is typically licensed under GPLv2 or CC-BY, making it perfectly suited for an OSS project.

- What it gives you: Lemmatization maps, true nouns, precise genders (m, f, n, c [commune/de-woord]), and regular/irregular plural forms.
- Application to your schema: This populates your base properties: lemma, partOfSpeech, gender, plural, and basic components of present/past verb groups.

### The Semantic & Grammar Goldmine: Dutch Wiktionary (Kaal / Raw Dumps)Wiktionary is under a CC-BY-SA license. Using a script to parse Wiktionary dumps and transform them into a completely new format (like your custom YAML structure) acts as a clean database derivation.

- What it gives you:
  - IPA Pronunciation: Dutch Wiktionary features highly standardized International Phonetic Alphabet strings and syllable structures (syllables, stress, ipa).
  - Verb Paradigms: Complete conjugation tables for your verb schema, specifically marking auxiliary verbs (hebben vs zijn), past singular/plural strings (liep/liepen), and perfect participles (gelopen).
  - Translations: Multi-language mapping keys (translations.en, translations.de).

### The Secret Weapon for CEFR Graded Content: NT2LexY

You should explicitly consider NT2Lex (Tack et al., 2018). It is a specialized, open lexical resource explicitly mapped out for Dutch as a Foreign Language (Nederlands als Tweede Taal).  

- What it gives you: It cross-references words with actual CEFR levels (A1 through C2) by analyzing text validation distributed across language-learning textbooks
- Licensing: Distributed via open academic licenses linked to Open Dutch WordNet.
- Application to your schema: Directly populates your cefr tag string.

### The Safe Frequency Alternative: OSCAR / Common Crawl or OpenSubtitles

Because SUBTLEX-NL is legally restricted for commercial distributions, you can substitute its frequency counts with raw counts derived from OpenSubtitles (OPUS corpus) or OSCAR / Common Crawl web-token lists.Run a filtering script to rank your OpenTaal lemmatized words against the OSCAR/OpenSubtitles raw frequency counts.This yields valid, custom numbers to map to your frequency.rank and frequency.occurrences targets without inheriting restrictive downstream claims.

### Scripting Priorities for Your Schema Elements:

1. The Noun Structure: Map lemmas from OpenTaal to look up target entries in Wiktionary. If the gender metadata equals o (onzijdig), assign n (neuter) to your schema. If it features v or m, assign your preferred gender label (c or u for the shared de-word category).
2. The Verb Structure: Irregular paradigms are difficult to parse algorithmically from scratch. Wiktionary templates use standardized conjugation tables (e.g., {{nl-verb-irreg}}). Target these tables directly to safely map properties like past.singular: liep.
3. CEFR Alignment: Inject the cefr variable from NT2Lex by looking up the primary dictionary key. If a word isn't listed in NT2Lex but sits in your top 1,000 frequencies, you can default-assign it an A1 reference weight for subsequent manual verification.

## Agent Notes
- Canonical Pydantic models live in `course_compiler/models.py` (language-agnostic,
  camelCase aliases, `to_yaml`). Conjugation tables are `{slot: form}` mappings so
  the schema isn't tied to Dutch pronouns.
- Dutch importer: `course_compiler/converters/dutch.py`. Sources:
  - kaikki.org JSONL (primary): pos, gender (`nl-noun` arg / sense tag),
    plural/diminutive forms (marked/archaic variants filtered), IPA (phonemic
    preferred), syllables (hyphenations), verb conjugations from form tags,
    English glosses → `translations.en`. Strong-verb detection via the `class`
    form tag sets `irregular`.
  - Open Dutch WordNet XML → synonyms by grouping lemmas per synset
    (`load_wordnet_synonyms`, ~31k lemmas, ~3s, iterparse+clear).
  - wordfreq cBpack → `Frequency` rank/zipf (`course_compiler/frequency.py`).
  - NT2Lex `.tsv` → `cefr` via `load_cefr_levels` (earliest attested CEFR level
    per lemma; 14,723 lemmas, A1–C1). Passed to both word/verb mappers.
  - Pure per-entry mappers (`word_from_kaikki`, `verb_from_kaikki`) +
    `convert_iterables` (no I/O) make it unit-testable; `convert` streams files.
- CLI: `course import --kaikki ... [--wordnet ...] [--frequency ...] [--nt2lex ...]
  --out courses/nl`.
- Verified on real data: 144k kaikki entries; smoke run of 8000 entries →
  4803 words + 1145 verbs, with synonyms + frequency + CEFR attached. Output NOT
  committed (large; generated under `courses/`).
- Full import verified: 87,480 words + 39,147 verbs written to `courses/nl`
  (file count matches entry count -- `_safe_name` appends a short stable hash when
  sanitizing alters the id, so distinct lemmas never overwrite each other).
- Known limitations / next: verb `auxiliary` (hebben/zijn) not extracted (not in
  the kaikki conjugation table); `stress` unset (could derive from IPA ˈ marker);
  ODWN synonyms can be noisy across synsets; homograph lemmas with multiple
  non-verb POS collapse to first seen (id=lemma).
- Deferred to a follow-up: the generic pluggable importer interface and
  CSV/TSV/YAML importers.
