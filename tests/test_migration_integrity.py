"""Upgrade-path migration integrity guards.

Defends against the v0.9.2 regression class: a stray or misnumbered migration
file sorting after the canonical RRF fix and silently overwriting it. See
CHANGELOG [0.9.2] for the full incident write-up.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "sql" / "migrations"


def _top_level_migrations() -> list[Path]:
    return sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.is_file())


def test_no_unnumbered_migrations():
    """Every top-level migration filename must start with a digit.

    `sql/upgrade.sh` applies migrations in alphabetical order. Any .sql file
    whose name begins with a non-digit character (e.g. ``update_search_function.sql``)
    sorts after numbered migrations and silently overrides them.
    """
    offenders = [p.name for p in _top_level_migrations() if not p.name[0].isdigit()]
    assert not offenders, (
        f"Unnumbered migration(s) found in sql/migrations/: {offenders}. "
        "Unnumbered files sort after numbered ones and override them on upgrade."
    )


def test_no_file_sorts_after_017():
    """No migration may sort alphabetically after 017_rrf_bm25.sql while
    breaking the RRF formula.

    017 is the RRF fix. A later migration may redefine hybrid_search_memories
    only if it preserves the canonical RRF pattern `1.0 / (rrf_k + coalesce(`
    AND does not reintroduce the broken raw-score fusion.
    """
    migrations = _top_level_migrations()
    rrf_fix = next((p for p in migrations if p.name == "017_rrf_bm25.sql"), None)
    assert rrf_fix is not None, "expected 017_rrf_bm25.sql at top-level sql/migrations/"

    later = [p.name for p in migrations if p.name > rrf_fix.name]
    offenders = []
    for name in later:
        content = (MIGRATIONS_DIR / name).read_text()
        if "hybrid_search_memories" not in content.lower():
            continue
        if "1.0 / (rrf_k + coalesce(" not in content:
            offenders.append(f"{name} (missing canonical RRF formula)")
            continue
        if "semantic_weight * coalesce(s.similarity" in content:
            offenders.append(f"{name} (reintroduces broken raw-score fusion)")

    assert not offenders, (
        f"Migration(s) sorting after 017_rrf_bm25.sql redefine hybrid_search_memories "
        f"without preserving true RRF: {offenders}"
    )


def test_017_rrf_bm25_is_functional_and_uses_rrf():
    """017 must contain a real RRF formula, not just a docs comment.

    The v0.8.3–v0.9.1 version of this file was comment-only. The v0.9.2 rewrite
    restores it to a functional migration with position-based RRF.
    """
    content = (MIGRATIONS_DIR / "017_rrf_bm25.sql").read_text()
    assert "create or replace function hybrid_search_memories" in content.lower(), (
        "017_rrf_bm25.sql must define hybrid_search_memories, not just document it"
    )
    assert "1.0 / (rrf_k + coalesce(" in content, (
        "017_rrf_bm25.sql must use true Reciprocal Rank Fusion: "
        "1.0 / (rrf_k + rank_ix), not raw-score linear combination"
    )
    broken_pattern = "semantic_weight * coalesce(s.similarity"
    assert broken_pattern not in content, (
        "017_rrf_bm25.sql contains the broken raw-score fusion pattern"
    )


def test_021_dim_aware_halfvec_is_dim_parametric():
    """021 must template halfvec casts via format() rather than hardcode dim.

    Kevin's request on issue #24: parse the column's format_type and template
    the cast so non-512 dims (1024, 3072, etc.) are first-class. A regression
    here would silently lock the migration to one dim again.
    """
    path = MIGRATIONS_DIR / "021_dim_aware_halfvec.sql"
    assert path.exists(), "expected sql/migrations/021_dim_aware_halfvec.sql"
    content = path.read_text()

    assert "format_type(a.atttypid, a.atttypmod)" in content, (
        "021 must introspect memories.embedding column type via format_type()"
    )
    assert "format(" in content and "halfvec(%1$s)" in content, (
        "021 must template halfvec casts using format() with a dim placeholder"
    )

    # No literal halfvec(512) / vector(512) in the migration body — every cast
    # must be templated. Comments are allowed to mention the literal.
    code_lines = [
        line for line in content.splitlines()
        if not line.lstrip().startswith("--")
    ]
    code = "\n".join(code_lines)
    assert "halfvec(512)" not in code, (
        "021 must not hardcode halfvec(512); template via format() instead"
    )
    assert "vector(512)" not in code, (
        "021 must not hardcode vector(512); template via format() instead"
    )

    # HNSW rebuild must stay opt-in via the documented session GUC.
    assert "ogham.rebuild_hnsw" in content, (
        "021 must gate HNSW index rebuild behind the ogham.rebuild_hnsw session GUC"
    )

    # Drop-prelude guard: enumerating pg_proc + dropping every overload by name
    # is what prevents a dim-change rerun (e.g. 512 -> 1024) from leaving
    # ambiguous overloads behind. Same incident class as the 9-vs-10-param
    # hybrid_search overload that 017 had to clean up.
    assert "pg_get_function_identity_arguments" in content, (
        "021 must enumerate existing overloads via pg_get_function_identity_arguments "
        "before recreating functions, otherwise dim-change reruns leave orphaned overloads"
    )
    assert "drop function if exists public" in content.lower(), (
        "021 must DROP each enumerated overload before CREATE OR REPLACE"
    )


def test_update_search_function_sql_does_not_exist():
    """The v0.9.1-era stray migration must stay removed."""
    stray = MIGRATIONS_DIR / "update_search_function.sql"
    assert not stray.exists(), (
        "sql/migrations/update_search_function.sql was removed in v0.9.2 "
        "because it silently overrode true RRF. Do not reintroduce it."
    )
