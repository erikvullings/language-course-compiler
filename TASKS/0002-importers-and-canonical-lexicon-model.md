# 0002 Importers and canonical lexicon model

Status: open
Priority: high
Owner: unassigned
Agent: unassigned
Area: pipeline
Depends on: 0001

## Context
Define the internal data model (word, verb, grammar, lesson, example, exercise)
and importer plugins that normalize external sources into it. The compiler must
contain no language-specific logic.

## Acceptance Criteria
- Canonical models matching the YAML schemas in `INITIAL_INSTRUCTIONS.md`
- At least CSV/TSV and YAML importers; pluggable importer interface
- Normalize + merge stages producing one model per lemma, indexed by id
- Reproducible: same inputs → identical output

## Implementation Notes
- Importers convert their source into the common internal model (one plugin per
  format: YAML, JSON, CSV, TSV, XML, Wiktionary, OpenTaal, WordNet, SUBTLEX, ...).
- Keep language config + datasets external (e.g. `courses/<lang>/`).

## Agent Notes
- Not started.
