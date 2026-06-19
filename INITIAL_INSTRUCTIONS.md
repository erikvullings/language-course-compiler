# Language Course Compiler

## Project Overview

Develop a language-agnostic course compiler that generates complete language-learning courses from lexical resources.

The compiler itself must contain **no Dutch-specific logic**. Dutch, German, French, Italian and Spanish are merely different configurations and input datasets.

The compiler should eventually support any language for which a sufficiently rich lexicon exists.

The output consists of two products:

1. Human-editable course sources (YAML + Markdown)
2. A single optimized JSON bundle consumed by the SPA.

The compiler is completely offline-first and reproducible. Running the compiler twice with the same inputs should generate identical output.

---

# Supported languages

Initially support:

* Dutch (nl)
* German (de)
* French (fr)
* Italian (it)
* Spanish (es)

The architecture must allow adding new languages without changing compiler code.

---

# Overall Pipeline

Input

Frequency list(s)
Lexical databases
Grammar definitions
Translation dictionaries
Optional images
Optional audio

↓

Import

↓

Normalize

↓

Merge

↓

Generate missing metadata

↓

Generate example sentences

↓

Generate grammar lessons

↓

Generate reading lessons

↓

Validate

↓

Generate exercises

↓

Generate audio

↓

Optimize

↓

Export

---

# Input Formats

The compiler should support importing lexical information from multiple formats.

Examples:

* YAML
* JSON
* CSV
* TSV
* XML
* Wiktionary dumps
* OpenTaal
* WordNet
* CELEX
* SUBTLEX
* OpenSubtitles
* custom frequency lists

Each importer converts its source into a common internal model.

---

# Canonical Lexicon Model

Every word eventually becomes a YAML document.

Example

```yaml
id: huis

language: nl

lemma: huis

normalized: huis

translations:
  en: house
  de: Haus
  fr: maison

partOfSpeech: noun

gender: n        # m, f, n, c, u

plural:
  regular: huizen
  alternatives: []

diminutive:
  regular: huisje

ipa: hœys

syllables:
  - huis

stress: 1

frequency:
  rank: 418
  occurrences: 38412
  source: subtlex

cefr: A1

audio:
  generated: audio/nl/words/huis.mp3
  recorded:

examples:
  - id: ex001
    nl: Ik woon in een groot huis.
    en: I live in a large house.
    de: Ich wohne in einem großen Haus.
    audio: audio/nl/examples/ex001.mp3

related:
  - woning
  - kamer

synonyms:

antonyms:

tags:
  - home
  - buildings

introducedInLesson: 12

reviewWeight: 1.0
```

---

# Verb Model

Verbs deserve their own schema.

```yaml
id: lopen

language: nl

lemma: lopen

translations:
  en: walk

auxiliary: hebben

infinitive: lopen

present:
  ik: loop
  jij: loopt
  u: loopt
  hij: loopt
  wij: lopen
  jullie: lopen
  zij: lopen

past:
  singular: liep
  plural: liepen

perfect:
  participle: gelopen

imperative:
  singular: loop
  plural: loopt

future:
  infinitive: lopen

conditional:

subjunctive:

irregular: true

audio:
  word: audio/nl/verbs/lopen.mp3
```

---

# Grammar Model

Grammar is generated independently from lessons.

Example

```yaml
id: present-tense

language: nl

title: Present Tense

cefr: A1

description:

rules:

examples:

exceptions:

relatedGrammar:

introducedInLesson:

exercises:
```

Grammar pages should later become Markdown.

---

# Lesson Generation

Lessons should NOT be written manually.

The compiler generates lessons using LLMs.

Each lesson introduces a configurable number of new words.

Example

Lesson 12

new words

* huis
* straat
* lopen

Allowed vocabulary

All previous words
+
current lesson words

The LLM must ONLY use allowed vocabulary.

---

# Vocabulary Validation

Every generated lesson is automatically validated.

Steps

1.

Tokenize

2.

Lemmatize

3.

Compare every lemma against

allowed vocabulary

If unknown words exist

Reject lesson.

Generate again.

The compiler must never generate lessons containing accidental vocabulary leakage.

---

# Grammar Progression

Grammar is introduced gradually.

The compiler maintains a dependency graph.

Example

Present tense

↓

Articles

↓

Plural nouns

↓

Word order

↓

Past tense

↓

Perfect tense

↓

Relative clauses

↓

Passive voice

↓

Subjunctive (if applicable)

Each grammar lesson contains

* explanation
* examples
* common mistakes
* exercises
* references to lessons
* references to words

Grammar explanations may be generated with an LLM but must be validated.

---

# Example Sentence Generation

Every lexical entry should receive several examples.

Rules

Use only previously introduced vocabulary whenever possible.

Generate

* beginner
* intermediate
* advanced

Translate into every supported interface language.

Generate audio.

---

# Exercise Generation

Generate exercises automatically.

Supported exercise types

Fill in the blank

Typing

Listening

Word ordering

Translation

Reverse translation

Multiple choice

Pronunciation

Matching

Flashcards

Conjugation

Grammar quizzes

Dictation

Reading comprehension

Exercises should reference lesson IDs and vocabulary IDs instead of duplicating data.

---

# Audio Generation

Support multiple providers.

Examples

Azure TTS

Google

OpenAI

Piper

Coqui

Prefer generating audio during compilation.

Store as MP3 or Opus.

Do NOT embed audio in JSON.

Instead store paths.

Example

audio/nl/words/huis.mp3

audio/nl/examples/ex001.mp3

---

# Offline Audio

The SPA should work offline.

Preferred approach

The browser caches audio progressively.

Recently used audio is available offline.

Fallback

Use browser Speech Synthesis API.

If cached audio exists

play recording

Else

browser TTS

This avoids embedding hundreds of MB inside the application.

---

# Images

Images are optional.

Store only paths.

Never embed binary data.

---

# Course Folder Structure

courses/

```
nl/

    words/

    verbs/

    grammar/

    lessons/

    examples/

    audio/

    images/

de/

fr/

it/

es/
```

---

# Build Commands

course build

Compile complete course

course validate

Run validation

course generate-audio

Generate all missing audio

course generate-lessons

Generate lessons

course generate-grammar

Generate grammar

course generate-exercises

Generate exercises

course import

Import lexical databases

course export

Export optimized JSON

course stats

Generate statistics

---

# Validation

Compiler should detect

missing translations

missing IPA

duplicate lemmas

duplicate IDs

missing audio

broken references

invalid lesson order

grammar dependency cycles

unknown vocabulary

unused words

unused grammar

missing examples

---

# JSON Export

The SPA should load one optimized JSON file.

Structure

{
metadata,

```
languages,

words,

verbs,

grammar,

lessons,

examples,

exercises,

indices
```

}

All objects should be indexed by ID.

Avoid duplication.

Lessons reference word IDs.

Exercises reference lesson IDs.

Grammar references word IDs.

---

# Suggested JSON Schema

{
"metadata": {
"courseLanguage": "nl",
"interfaceLanguages": [
"en",
"de",
"fr"
],
"version": "1.0",
"generatedAt": "",
"compilerVersion": ""
},

"words": {
"huis": {
"...": "..."
}
},

"verbs": {
"lopen": {
"...": "..."
}
},

"grammar": {
"present-tense": {
"...": "..."
}
},

"lessons": [
{
"id": "lesson001",

```
  "title": "...",

  "introducedWords": [
    "huis",
    "lopen"
  ],

  "grammar": [
    "present-tense"
  ],

  "text": "...",

  "audio": "...",

  "exercises": [
    "exercise001"
  ]
}
```

],

"examples": {
"ex001": {}
},

"exercises": {
"exercise001": {}
},

"indices": {
"wordsByFrequency": [],
"wordsByLesson": [],
"grammarByLesson": [],
"examplesByWord": {}
}
}

---

# Future Architecture

The compiler should be modular.

Each pipeline stage should be an independent plugin.

Examples

Importer Plugin

Exporter Plugin

Lesson Generator

Grammar Generator

Audio Generator

Validator

Image Generator

Translation Provider

TTS Provider

LLM Provider

New providers should be configurable without changing compiler code.

---

# Long-Term Goal

The project should become a reusable language-course compiler capable of generating complete, structured, high-quality language-learning courses for any language from open lexical resources.

The generated output should support:

* progressive vocabulary acquisition
* grammar instruction
* interactive exercises
* spaced repetition
* pronunciation
* audio
* offline learning
* multilingual interface translations

while remaining completely static, reproducible, version-controlled and suitable for deployment as a PWA on GitHub Pages.

Instead of generating a single JSON output, generate a manifest plus multiple JSON bundles. For example:

* manifest.json (course metadata and version)
* words.json (all lexical entries)
* verbs.json
* grammar.json
* lessons/lesson001.json, lesson002.json, …
* exercises.json
