"""Language-dependent importers that map source data onto the canonical models.

Each module here knows the quirks of one language's source datasets. Keeping
them separate from :mod:`course_compiler.models` preserves the rule that the
canonical schema (and the rest of the compiler) stays language-agnostic.
"""
