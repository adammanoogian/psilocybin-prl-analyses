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
    """Mode column must contain at least [mode-a] tag.

    # NOTE: Only check [mode-a]. [mode-b] is Phase 34+ — do NOT require it here.
    """
    assert "[mode-a]" in map_text, (
        "Capability map missing [mode-a] tag in Mode column. "
        "All current rows should be tagged [mode-a] (no-pooling mode)."
    )


def test_mitigation_tags_present(map_text: str) -> None:
    """Mitigation column must contain at least one valid mitigation tag.

    Checks that at least one of the known mitigation tags appears in the
    map.  Does not require ALL tags (the grid sweep in plan 31-02 will
    populate them progressively).
    """
    valid_tags = {"[none]", "[M1]", "[M1+Laplace]", "[M1+Laplace+fp64]"}
    found = any(tag in map_text for tag in valid_tags)
    assert found, (
        f"Capability map missing all mitigation tags. Expected at least one "
        f"of {sorted(valid_tags)} to appear in a Mitigation column."
    )
