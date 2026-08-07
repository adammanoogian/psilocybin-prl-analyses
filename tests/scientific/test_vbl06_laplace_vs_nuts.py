"""VBL-06: Laplace-vs-NUTS posterior comparison harness for PAT-RL.

Loads two ``az.InferenceData`` (one Laplace, one NUTS) fit on the same
``sim_df`` and produces a per-subject per-parameter diff table.

Tolerance gates from quick-004 VB_LAPLACE_FEASIBILITY.md Section 6:
- |Δ posterior_mean(omega_2)| < 0.3
- |Δ log_sd(omega_2)| < 0.5

Soft warning (log-only): sd_laplace / sd_nuts outside [0.5, 2.0] for
any parameter.

Mirrors the structural convention of
``tests/scientific/test_valid03_cross_platform.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific

# Make the project root and src/ importable when running this file directly.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if str(Path(_project_root) / "src") not in sys.path:
    sys.path.insert(0, str(Path(_project_root) / "src"))

import arviz as az  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)

# Tolerance gates from VB_LAPLACE_FEASIBILITY.md Section 6
_GATE_ABS_DIFF_MEAN_OMEGA2: float = 0.3
_GATE_ABS_DIFF_LOG_SD_OMEGA2: float = 0.5
_SOFT_SD_RATIO_LOWER: float = 0.5
_SOFT_SD_RATIO_UPPER: float = 2.0


# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------


def compare_posteriors(
    idata_laplace: az.InferenceData,
    idata_nuts: az.InferenceData,
    params: tuple[str, ...] = ("omega_2", "beta"),
) -> pd.DataFrame:
    """Per-subject per-parameter diff table.

    Parameters
    ----------
    idata_laplace : az.InferenceData
        Laplace InferenceData (from ``fit_vb_laplace_patrl``).  Must have
        ``participant_id`` coord in ``posterior``.
    idata_nuts : az.InferenceData
        NUTS InferenceData (from ``fit_batch_hierarchical_patrl``).  May
        use either ``participant_id`` or ``participant`` as the coord name;
        if ``participant`` is detected, it is renamed on a `.copy()` before
        comparison.  The caller's ``idata_nuts`` is **never mutated**.
    params : tuple[str, ...], default ('omega_2', 'beta')
        Parameter names to compare.  Must be present in BOTH idatas'
        ``posterior`` groups.

    Returns
    -------
    pd.DataFrame
        Long format: one row per (participant_id, parameter). Columns:
        ``participant_id``, ``parameter``, ``mean_laplace``, ``mean_nuts``,
        ``sd_laplace``, ``sd_nuts``, ``abs_diff_mean``, ``abs_diff_log_sd``,
        ``sd_ratio``, ``within_gate``.

        The ``within_gate`` column is ``True`` iff the row is for
        ``omega_2`` AND ``abs_diff_mean < 0.3`` AND
        ``abs_diff_log_sd < 0.5``.  Non-omega_2 rows (e.g. ``beta``)
        receive ``pd.NA`` for ``within_gate`` — those rows are informational
        only (no tolerance defined for them in Phase 19; see
        VB_LAPLACE_FEASIBILITY.md Section 6).
        Column dtype: ``pd.BooleanDtype()`` (nullable bool).

    Notes
    -----
    ``idata_nuts`` is treated as read-only; this function takes a `.copy()`
    before renaming.  The rename is scoped to the comparison: the raw NUTS
    idata at the caller side still uses dim ``'participant'`` and will NOT
    work with ``export_subject_trajectories`` until ``_samples_to_idata`` is
    fixed (tracked as OQ1 follow-up in STATE.md).

    Raises
    ------
    RuntimeError
        If the NUTS coord rename fails, or if the two idatas do not share
        the same participant set.
    """
    # ------------------------------------------------------------------
    # 1. Detect coord names and normalise NUTS idata if needed
    # ------------------------------------------------------------------
    lap_posterior = idata_laplace.posterior  # type: ignore[attr-defined]
    nuts_posterior = idata_nuts.posterior  # type: ignore[attr-defined]

    # Laplace always emits 'participant_id'; fall back to 'participant'
    # if an unusual idata is passed (should not happen in practice).
    coord_name_lap = (
        "participant_id" if "participant_id" in lap_posterior.coords else "participant"
    )

    coord_name_nuts = (
        "participant_id"
        if "participant_id" in nuts_posterior.coords
        else "participant"
    )

    if coord_name_nuts != "participant_id":
        logger.warning(
            "NUTS idata uses dim %r not 'participant_id'; renamed on a "
            "copy for this comparison only. See STATE.md OQ1 follow-up "
            "to fix _samples_to_idata in a separate hotfix.",
            coord_name_nuts,
        )
        try:
            idata_nuts = idata_nuts.copy()
            idata_nuts = idata_nuts.rename({coord_name_nuts: "participant_id"})
            nuts_posterior = idata_nuts.posterior  # type: ignore[attr-defined]
            coord_name_nuts = "participant_id"
        except Exception as exc:
            raise RuntimeError(
                f"Failed to rename NUTS idata coord from {coord_name_nuts!r} to "
                f"'participant_id'. Expected coord name: 'participant_id', "
                f"actual: {coord_name_nuts!r}. "
                f"Underlying error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # 2. Extract and validate participant sets
    # ------------------------------------------------------------------
    pids_lap = sorted(str(v) for v in lap_posterior.coords[coord_name_lap].values)
    pids_nuts = sorted(str(v) for v in nuts_posterior.coords["participant_id"].values)

    if pids_lap != pids_nuts:
        raise RuntimeError(
            f"Participant set mismatch between Laplace and NUTS idatas. "
            f"Expected (Laplace): {pids_lap}. "
            f"Actual (NUTS):      {pids_nuts}."
        )

    pids = pids_lap  # canonical ordered list

    # ------------------------------------------------------------------
    # 3. Build diff rows
    # ------------------------------------------------------------------
    rows: list[dict] = []

    for param in params:
        for pid in pids:
            # --- Laplace ---
            mean_lap = float(
                lap_posterior[param]
                .sel({coord_name_lap: pid})
                .mean()
                .values
            )
            sd_lap = float(
                lap_posterior[param]
                .sel({coord_name_lap: pid})
                .std()
                .values
            )

            # --- NUTS ---
            mean_nuts = float(
                nuts_posterior[param]
                .sel(participant_id=pid)
                .mean()
                .values
            )
            sd_nuts = float(
                nuts_posterior[param]
                .sel(participant_id=pid)
                .std()
                .values
            )

            # --- Derived metrics ---
            abs_diff_mean = abs(mean_lap - mean_nuts)

            # Guard: near-zero sd
            if sd_lap <= 0 or sd_nuts <= 0:
                logger.warning(
                    "Near-zero sd detected: participant=%s param=%s "
                    "sd_laplace=%g sd_nuts=%g; setting abs_diff_log_sd=inf.",
                    pid,
                    param,
                    sd_lap,
                    sd_nuts,
                )
                abs_diff_log_sd = np.inf
            else:
                abs_diff_log_sd = abs(np.log(sd_lap) - np.log(sd_nuts))

            # sd_ratio
            if sd_nuts <= 0:
                sd_ratio = np.inf
            else:
                sd_ratio = sd_lap / sd_nuts

            rows.append(
                {
                    "participant_id": pid,
                    "parameter": param,
                    "mean_laplace": mean_lap,
                    "mean_nuts": mean_nuts,
                    "sd_laplace": sd_lap,
                    "sd_nuts": sd_nuts,
                    "abs_diff_mean": abs_diff_mean,
                    "abs_diff_log_sd": abs_diff_log_sd,
                    "sd_ratio": sd_ratio,
                }
            )

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Build within_gate column (nullable bool)
    # ------------------------------------------------------------------
    df["within_gate"] = pd.array([pd.NA] * len(df), dtype=pd.BooleanDtype())

    omega2_mask = df["parameter"] == "omega_2"
    if omega2_mask.any():
        gate_vals = (
            df.loc[omega2_mask, "abs_diff_mean"] < _GATE_ABS_DIFF_MEAN_OMEGA2
        ) & (
            df.loc[omega2_mask, "abs_diff_log_sd"] < _GATE_ABS_DIFF_LOG_SD_OMEGA2
        )
        df.loc[omega2_mask, "within_gate"] = pd.array(
            gate_vals.tolist(), dtype=pd.BooleanDtype()
        )

    return df


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------


def _apply_hard_gates(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Apply hard tolerance gates and soft SD-ratio warnings.

    Hard gates apply only to ``omega_2`` rows (per VB_LAPLACE_FEASIBILITY.md
    Section 6).  Non-``omega_2`` rows retain ``pd.NA`` in ``within_gate``
    and are NEVER consulted for the hard gate decision.

    Soft SD-ratio warnings scan every row and are informational only.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare_posteriors`.

    Returns
    -------
    all_pass : bool
        ``True`` iff every ``omega_2`` row passes both hard gate conditions.
    messages : list[str]
        Human-readable gate/warning messages.  ``HARD FAIL:`` prefix for
        gate violations; ``SOFT WARN:`` prefix for sd_ratio violations.
    """
    omega2 = df[df["parameter"] == "omega_2"]

    # within_gate on omega_2 rows is guaranteed non-null by compare_posteriors.
    all_pass = bool(omega2["within_gate"].all())

    messages: list[str] = []

    for _, row in omega2.iterrows():
        if row["abs_diff_mean"] >= _GATE_ABS_DIFF_MEAN_OMEGA2:
            messages.append(
                f"HARD FAIL: participant={row['participant_id']} "
                f"abs_diff_mean(omega_2)={row['abs_diff_mean']:.3f} "
                f">= gate {_GATE_ABS_DIFF_MEAN_OMEGA2}"
            )
        if row["abs_diff_log_sd"] >= _GATE_ABS_DIFF_LOG_SD_OMEGA2:
            messages.append(
                f"HARD FAIL: participant={row['participant_id']} "
                f"abs_diff_log_sd(omega_2)={row['abs_diff_log_sd']:.3f} "
                f">= gate {_GATE_ABS_DIFF_LOG_SD_OMEGA2}"
            )

    # Soft SD-ratio warnings scan every row (all parameters).
    for _, row in df.iterrows():
        if not (_SOFT_SD_RATIO_LOWER <= row["sd_ratio"] <= _SOFT_SD_RATIO_UPPER):
            messages.append(
                f"SOFT WARN: participant={row['participant_id']} "
                f"parameter={row['parameter']} sd_ratio={row['sd_ratio']:.3f} "
                f"outside [{_SOFT_SD_RATIO_LOWER}, {_SOFT_SD_RATIO_UPPER}]"
            )

    return all_pass, messages


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point for VBL-06.

    Returns
    -------
    int
        Exit code: 0 if all omega_2 gates pass (or ``--skip-nuts-comparison``
        flag set), 1 if any omega_2 gate fails, 2 on loader/import error.
    """
    parser = argparse.ArgumentParser(
        prog="vbl06_laplace_vs_nuts.py",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # -- run subcommand -------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help=(
            "Run Laplace fit on sim_df; optionally also run NUTS for "
            "comparison. Saves both idatas to --output-dir."
        ),
    )
    p_run.add_argument(
        "--sim-df",
        type=Path,
        required=True,
        help="Path to parquet sim_df.",
    )
    p_run.add_argument(
        "--model",
        type=str,
        default="hgf_2level_patrl",
        help="Model name (default: hgf_2level_patrl).",
    )
    p_run.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write Laplace (and optionally NUTS) idata .nc files.",
    )
    p_run.add_argument(
        "--skip-nuts-comparison",
        action="store_true",
        help=(
            "Only run Laplace fit; skip NUTS comparison. "
            "Exits 0. Useful while cluster NUTS results are pending."
        ),
    )
    p_run.add_argument(
        "--params",
        type=str,
        nargs="+",
        default=["omega_2", "beta"],
        help="Parameters to compare (default: omega_2 beta).",
    )

    # -- compare subcommand ---------------------------------------------------
    p_cmp = sub.add_parser(
        "compare",
        help="Load two saved idatas and produce a diff table.",
    )
    p_cmp.add_argument(
        "--laplace",
        type=Path,
        required=True,
        help="Path to Laplace idata .nc file.",
    )
    p_cmp.add_argument(
        "--nuts",
        type=Path,
        required=True,
        help="Path to NUTS idata .nc file.",
    )
    p_cmp.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        dest="output",
        help="Optional path to write diff CSV.",
    )
    p_cmp.add_argument(
        "--params",
        type=str,
        nargs="+",
        default=["omega_2", "beta"],
        help="Parameters to compare (default: omega_2 beta).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[VBL-06] %(levelname)s %(message)s",
    )

    if args.cmd == "run":
        try:
            import pandas as pd  # noqa: F811

            from prl_hgf.fitting.fit_vb_laplace_patrl import fit_vb_laplace_patrl
        except ImportError as exc:
            logger.error("Import error loading fit_vb_laplace_patrl: %s", exc)
            return 2

        try:
            sim_df = pd.read_parquet(str(args.sim_df))
        except Exception as exc:
            logger.error("Failed to load sim_df from %s: %s", args.sim_df, exc)
            return 2

        logger.info("Running Laplace fit on %s ...", args.sim_df)
        idata_lap = fit_vb_laplace_patrl(sim_df, args.model)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        lap_path = args.output_dir / "laplace.nc"
        idata_lap.to_netcdf(str(lap_path))
        logger.info("Saved Laplace idata to %s", lap_path)

        if args.skip_nuts_comparison:
            logger.info("NUTS comparison skipped per --skip-nuts-comparison flag.")
            return 0

        # Run NUTS comparison (deferred import keeps this optional)
        try:
            from prl_hgf.fitting.hierarchical_patrl import (
                fit_batch_hierarchical_patrl,
            )
        except ImportError as exc:
            logger.error("Import error loading hierarchical_patrl: %s", exc)
            return 2

        logger.info("Running NUTS fit for comparison ...")
        idata_nuts = fit_batch_hierarchical_patrl(sim_df, args.model)
        nuts_path = args.output_dir / "nuts.nc"
        idata_nuts.to_netcdf(str(nuts_path))
        logger.info("Saved NUTS idata to %s", nuts_path)

        df = compare_posteriors(idata_lap, idata_nuts, tuple(args.params))
        all_pass, messages = _apply_hard_gates(df)
        for msg in messages:
            if msg.startswith("HARD FAIL"):
                logger.error(msg)
            else:
                logger.warning(msg)
        return 0 if all_pass else 1

    else:  # compare subcommand
        try:
            idata_lap = az.from_netcdf(str(args.laplace))
            idata_nuts = az.from_netcdf(str(args.nuts))
        except Exception as exc:
            logger.error("Failed to load idata files: %s", exc)
            return 2

        df = compare_posteriors(idata_lap, idata_nuts, tuple(args.params))

        if args.output is not None:
            df.to_csv(str(args.output), index=False)
            logger.info("Wrote diff CSV to %s", args.output)

        # Print table summary
        print(df.to_string(index=False))

        all_pass, messages = _apply_hard_gates(df)
        for msg in messages:
            if msg.startswith("HARD FAIL"):
                logger.error(msg)
            else:
                logger.warning(msg)

        return 0 if all_pass else 1


def test_hard_gates_pass_on_within_tolerance() -> None:
    """Trivial smoke: differences below tolerance pass the hard gates."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "subject": [0, 1],
            "param": ["omega_2", "omega_2"],
            "mean_laplace": [0.5, 0.6],
            "mean_nuts": [0.55, 0.65],
            "diff_mean": [0.05, 0.05],
            "log_sd_laplace": [0.1, 0.1],
            "log_sd_nuts": [0.1, 0.1],
            "diff_log_sd": [0.0, 0.0],
        }
    )
    all_pass, _ = _apply_hard_gates(df)
    assert all_pass


if __name__ == "__main__":
    sys.exit(main())
