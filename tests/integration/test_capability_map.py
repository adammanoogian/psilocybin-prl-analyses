"""Closure-guard tests for ``docs/CAPABILITY_MAP.md``.

The capability map is the source-of-truth record of what HGF fitting
configurations have been empirically validated in this repo. These tests
do NOT enforce that every fit run updates the map (no automated way to
verify that without a separate run-registry); instead they enforce
*structural* integrity so the map stays parseable and reviewable:

- Map file exists and is non-trivial.
- All required columns appear in every results table.
- The "How to update" section exists so future maintainers see the protocol.
- Pointers to companion docs and memory files are present.
- No row left with literal placeholder text (``TBD``, ``XXX``, ``???``)
  in a status cell that claims to be PASS or FAIL.

Update the map at ``docs/CAPABILITY_MAP.md`` whenever a fit run produces a
new data point. See the "How to update" section in that file for the
contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAP_PATH = REPO_ROOT / "docs" / "CAPABILITY_MAP.md"

# Columns every results table in the map must declare. Order is not
# enforced (a future map could re-order); presence is.
REQUIRED_COLUMNS = {
    "Model",
    "Sampler",
    "Mass-Matrix",
    "Mode",
    "Mitigation",
    "P_total",
    "Status",
    "Walltime",
    "Evidence",
    "Diagnostics",
}

# Sections that must exist by name. The map's structure can evolve, but
# these anchors are what consumers (sister repos, AI agents, future
# maintainers) navigate to.
REQUIRED_SECTIONS = (
    "## Conventions",
    "## Capability table",
    "## Open gaps",
    "## Planned runs",
    "## Mitigation candidates",
    "## How to update",
    "## Source-of-truth notes",
)

# Companion docs / memory files that the map references. If any of these
# move, the map's pointers go stale silently — this test catches that.
REQUIRED_REFERENCES = (
    "memory/project_end_goal_capability_map.md",
    "memory/project_phase25_shelved.md",
    "docs/PAT_RL_PHASE20_HANDOFF.md",
    "docs/PHASE_14_1_07_VERIFICATION.md",
)

# Placeholder tokens that must NOT appear in a row marked PASS or FAIL.
# (Allowed in PENDING / NOT TESTED rows because those rows haven't been
# populated yet.)
PLACEHOLDER_PATTERNS = (r"\bTBD\b", r"\bXXX\b", r"\?\?\?")


@pytest.fixture(scope="module")
def map_text() -> str:
    """Read the capability map once per test module."""
    assert MAP_PATH.is_file(), f"Capability map missing: {MAP_PATH}"
    return MAP_PATH.read_text(encoding="utf-8")


def test_capability_map_exists_and_nontrivial(map_text: str) -> None:
    """Map file present and has substantive content."""
    assert len(map_text) > 1000, (
        f"docs/CAPABILITY_MAP.md is {len(map_text)} chars — too short to "
        f"contain a useful table. Did the file get truncated?"
    )


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_required_section_present(map_text: str, section: str) -> None:
    """All structural anchor sections must be present by name."""
    assert section in map_text, (
        f"Capability map missing required section: {section!r}.\n"
        f"Sections give consumers stable navigation anchors. If you "
        f"renamed a section, update REQUIRED_SECTIONS in this test."
    )


@pytest.mark.parametrize("column", sorted(REQUIRED_COLUMNS))
def test_required_column_in_at_least_one_table(map_text: str, column: str) -> None:
    """Every required column appears in at least one table header line."""
    table_header_pattern = re.compile(r"^\|.*\|$", re.MULTILINE)
    headers = table_header_pattern.findall(map_text)
    assert any(column in h for h in headers), (
        f"Capability map: no table header contains required column "
        f"{column!r}. Did the table get reformatted?"
    )


@pytest.mark.parametrize("ref", REQUIRED_REFERENCES)
def test_companion_reference_resolves(map_text: str, ref: str) -> None:
    """Cross-doc and cross-memory pointers must (a) appear in the map and
    (b) point at files that exist on disk."""
    assert ref in map_text, (
        f"Capability map should reference {ref!r} (cross-doc/cross-memory "
        f"navigation). If the reference moved, update both the map and "
        f"REQUIRED_REFERENCES in this test."
    )
    # Memory files live outside the repo (under ~/.claude/...), so only
    # check repo-relative refs on disk.
    if ref.startswith("docs/"):
        target = REPO_ROOT / ref
        assert target.exists(), (
            f"Capability map references {ref!r} but that file does not "
            f"exist on disk. Either restore the file or update the map."
        )


def test_no_placeholder_in_completed_rows(map_text: str) -> None:
    """Rows marked PASS or FAIL must not contain TBD/XXX/??? placeholders.

    PENDING / NOT TESTED rows are permitted to have placeholders since
    their data hasn't been collected yet.
    """
    completed_row_re = re.compile(r"^\|.*(?:✅ PASS|❌ FAIL|❌ TIMEOUT).*\|$", re.MULTILINE)
    completed_rows = completed_row_re.findall(map_text)
    assert completed_rows, (
        "Capability map has zero completed (PASS / FAIL / TIMEOUT) rows. "
        "If this is intentional during early scaffolding, expand this test."
    )
    for row in completed_rows:
        for pattern in PLACEHOLDER_PATTERNS:
            assert not re.search(pattern, row), (
                f"Completed row in capability map still has placeholder "
                f"text matching {pattern!r}:\n  {row}\n"
                f"Either fill in the real value or downgrade the status."
            )


def test_pending_rows_have_evidence_pointer(map_text: str) -> None:
    """🔲 PENDING rows must point at a planned/in-flight job so we can
    track which runs will populate them."""
    pending_row_re = re.compile(r"^\|.*🔲 PENDING.*\|$", re.MULTILINE)
    for row in pending_row_re.findall(map_text):
        # Either a job ID (digits) or "overnight job" / "queued" / commit hash
        has_job_pointer = bool(
            re.search(r"job\s+\d+", row)
            or re.search(r"overnight job", row, re.IGNORECASE)
            or re.search(r"queued", row, re.IGNORECASE)
            or re.search(r"commit\s+`[0-9a-f]{6,}`", row)
        )
        assert has_job_pointer, (
            f"PENDING row missing evidence pointer (need a job ID, "
            f"'overnight job', 'queued', or commit hash so we can trace "
            f"which run will populate it):\n  {row}"
        )


def test_mass_matrix_tags_present(map_text: str) -> None:
    """Mass-Matrix column must contain at least [diagonal] and [dense] tags.

    These are the two empirically tested mass matrix configurations.
    [diagonal] is the default; [dense] tracks the BlackJAX 1.5 dense run.
    """
    assert "[diagonal]" in map_text, (
        "Capability map missing [diagonal] tag in Mass-Matrix column. "
        "All standard NUTS rows should be tagged [diagonal]."
    )
    assert "[dense]" in map_text, (
        "Capability map missing [dense] tag in Mass-Matrix column. "
        "The BlackJAX 1.5 dense mass-matrix row (DEPS-05 evidence) "
        "should be tagged [dense]."
    )


def test_mode_tags_present(map_text: str) -> None:
    """Mode column must contain [mode-a] and [mode-b] tags.

    [mode-a] covers no-pooling runs (Phase 31 grid sweep).
    [mode-b] covers hierarchical-pooling runs (Phase 34 grid sweep).
    Both modes are required once the Phase 34 grid sweep is complete.

    NOTE: [mode-b] check is deferred until Phase 34-02 populates rows.
    Only [mode-a] is enforced here for now.
    """
    assert "[mode-a]" in map_text, (
        "Capability map missing [mode-a] tag in Mode column. "
        "All current rows should be tagged [mode-a] (no-pooling mode)."
    )


def test_mitigation_tags_present(map_text: str) -> None:
    """Mitigation column must contain at least one valid mitigation tag.

    Checks that at least one of the known mitigation tags appears in the
    map.  Does not require ALL tags (the grid sweeps in plans 31-02 and
    34-02 will populate them progressively).

    Valid tags include both Mode A (no-pooling) and Mode B (hierarchical)
    mitigation variants.
    """
    valid_tags = {
        "[none]",
        "[M1]",
        "[M1+Laplace]",
        "[M1+Laplace+fp64]",
        "[hier+M1]",
        "[hier+M1+M2]",
        "[hier+M1+M2+Laplace]",
        "[hier+M1+M2+Laplace+covariates]",
    }
    found = any(tag in map_text for tag in valid_tags)
    assert found, (
        f"Capability map missing all mitigation tags. Expected at least one "
        f"of {sorted(valid_tags)} to appear in a Mitigation column."
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Phase 31 grid sweep incomplete — only 7 of 48 cells have results "
        "(all TIMEOUT/CRASH). Remaining cells need cuda13 resubmission."
    ),
)
def test_all_mode_a_mitigation_combos_present(map_text: str) -> None:
    """All 4 Mode A mitigation combo tags must appear in the map.

    Tags: [none], [M1], [M1+Laplace], [M1+Laplace+fp64].
    Enforced after Phase 31 grid sweep populates results.
    """
    required_tags = {"[none]", "[M1]", "[M1+Laplace]", "[M1+Laplace+fp64]"}
    mode_a_rows = re.findall(
        r"^\|.*\[mode-a\].*\|$", map_text, re.MULTILINE
    )
    found_tags = set()
    for row in mode_a_rows:
        for tag in required_tags:
            if tag in row:
                found_tags.add(tag)
    missing = required_tags - found_tags
    assert not missing, (
        f"Capability map Mode A rows missing mitigation tags: {sorted(missing)}. "
        f"Found {sorted(found_tags)} in {len(mode_a_rows)} Mode A rows."
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Phase 31 grid sweep incomplete — only 7 valid Mode A rows. "
        "Need cuda13 resubmission for remaining 41 cells."
    ),
)
def test_mode_a_minimum_coverage(map_text: str) -> None:
    """At least 24 Mode A rows with completed status must exist."""
    mode_a_completed = re.findall(
        r"^\|.*\[mode-a\].*(?:PASS|FAIL|TIMEOUT|CRASH|INVALID).*\|$",
        map_text,
        re.MULTILINE,
    )
    assert len(mode_a_completed) >= 24, (
        f"Capability map has {len(mode_a_completed)} completed Mode A rows, "
        f"expected at least 24 (2 models x 6 n_per_group x 2 mitigations min)."
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "MODEA-08 not met — no mitigation combo cleared the 3-level cliff "
        "at P=300. All attempted cells TIMEOUT or CRASH."
    ),
)
def test_cliff_cleared_row_exists(map_text: str) -> None:
    """At least one 3-level row at P=300 must show PASS with mitigations.

    This is the MODEA-08 closure guard: empirical proof that the mitigation
    ladder clears the 3-level conditioning cliff at production cohort size.
    """
    cliff_rows = re.findall(
        r"^\|.*3-level.*300.*(?:✅ PASS).*(?:\[M1\+Laplace\]|\[M1\+Laplace\+fp64\]).*\|$",
        map_text,
        re.MULTILINE,
    )
    assert len(cliff_rows) >= 1, (
        "No 3-level PASS row at P=300 with M1+Laplace or M1+Laplace+fp64. "
        "MODEA-08 requires at least one such row. "
        "See MODEA-08 Status section in CAPABILITY_MAP.md."
    )


@pytest.mark.skipif(
    # TODO(phase-34-03): remove skipif after grid sweep results are populated
    not bool(
        __import__("re").search(
            r"^\|.*\[mode-b\].*(?:PASS|FAIL|TIMEOUT|CRASH|INVALID)",
            __import__("pathlib").Path(__file__).resolve().parents[2]
            .joinpath("docs", "CAPABILITY_MAP.md")
            .read_text(encoding="utf-8")
            if __import__("pathlib").Path(__file__).resolve().parents[2]
            .joinpath("docs", "CAPABILITY_MAP.md")
            .is_file()
            else "",
            __import__("re").MULTILINE,
        )
    ),
    reason="No completed [mode-b] data rows in capability map yet — skipping until Phase 34-02 runs",
)
def test_mode_b_minimum_cells(map_text: str) -> None:
    """At least 24 Mode B rows must exist with completed status."""
    mode_b_completed = re.findall(
        r"^\|.*\[mode-b\].*(?:PASS|FAIL|TIMEOUT|CRASH|INVALID).*\|$",
        map_text,
        re.MULTILINE,
    )
    assert len(mode_b_completed) >= 24, (
        f"Capability map has {len(mode_b_completed)} completed Mode B rows, "
        f"expected at least 24 (2 models x 3 n_per_group x 4 mitigations)."
    )
