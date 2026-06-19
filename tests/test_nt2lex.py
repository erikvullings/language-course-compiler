"""NT2Lex CEFR level derivation."""

from __future__ import annotations

from course_compiler.converters import dutch

# Minimal NT2Lex-shaped TSV: only the columns the loader needs must line up by
# header name, so we include the F@<level> columns it looks for.
HEADER = "word\ttag\tsense_se-id\tF@A1\tF@A2\tF@B1\tF@B2\tF@C1"


def _write(tmp_path, rows):
    path = tmp_path / "nt2lex.tsv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


def test_word_level_is_earliest_attested(tmp_path):
    # huis appears already at A1; abstract only from B2
    path = _write(
        tmp_path,
        [
            "huis\tN\thuis-n-1\t9\t37\t26\t-\t-",
            "abstract\tADJ\tabstract-a-1\t-\t-\t-\t4\t2",
        ],
    )
    levels = dutch.load_cefr_levels(path)
    assert levels["huis"] == "A1"
    assert levels["abstract"] == "B2"


def test_minimum_level_across_senses_wins(tmp_path):
    # Two senses of the same lemma; the earliest level for the lemma is A2.
    path = _write(
        tmp_path,
        [
            "blad\tN\tblad-n-1\t-\t12\t5\t-\t-",
            "blad\tN\tblad-n-2\t-\t-\t3\t1\t-",
        ],
    )
    levels = dutch.load_cefr_levels(path)
    assert levels["blad"] == "A2"


def test_word_with_no_attestation_is_absent(tmp_path):
    path = _write(tmp_path, ["leeg\tN\tleeg-n-1\t-\t-\t-\t-\t-"])
    levels = dutch.load_cefr_levels(path)
    assert "leeg" not in levels
