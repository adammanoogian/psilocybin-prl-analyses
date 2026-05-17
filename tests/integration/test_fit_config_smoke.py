"""Smoke tests for ``fit_batch_hierarchical`` configuration matrix.

Regression guard for Phase 28 refactor: exercises the cartesian product of
fitting flags (model_name x use_laplace_warmup x tight_omega3_prior) against
the CURRENT pre-refactor API.  After the FitConfig refactor lands, these
tests will be updated to use the new dataclass interface while preserving
the same behavioral assertions.

Each parametrized cell:
1. Generates synthetic data via the existing simulation pipeline.
2. Calls ``fit_batch_hierarchical`` with minimal MCMC settings (2 chains,
   50 draws, 50 warmup) so the test exercises the full code path without
   requiring meaningful convergence.
3. Asserts: no exception, result contains ArviZ InferenceData with a
   posterior group.

Run::

    pytest tests/integration/test_fit_config_smoke.py -v -m integration
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable regardless of install mode.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Session-scoped fixture: small synthetic cohort DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sim_df_small():
    """Generate a small synthetic cohort (~4 participants).

    Uses ``n_per_group=2`` (1 group x 1 session = 2 participants) with the
    full simulation pipeline to produce realistic trial-level data.  Fixed
    seed ensures reproducibility across runs.

    The fixture is session-scoped so the (relatively expensive) simulation
    is shared across all parametrized test cells.
    """
    import pandas as pd

    from prl_hgf.env.simulator import generate_session
    from prl_hgf.env.task_config import load_config
    from prl_hgf.models.hgf_2level import build_2level_network
    from prl_hgf.simulation.agent import simulate_agent

    cfg = load_config()
    n_per_group = 2
    n_trials_max = 80  # Short sessions for speed
    base_seed = 88800
    rows: list[dict] = []

    for p_idx in range(n_per_group):
        pid = f"SMOKE{p_idx + 1:03d}"
        seed = base_seed + p_idx
        rng = np.random.default_rng(seed)
        net = build_2level_network(omega_2=-3.0)
        trials = generate_session(cfg, seed=seed)
        result = simulate_agent(
            net, trials, beta=3.0, zeta=0.5, rng=rng
        )

        for t_idx in range(min(n_trials_max, len(trials))):
            rows.append(
                {
                    "participant_id": pid,
                    "group": "control",
                    "session": "baseline",
                    "trial": t_idx,
                    "cue_chosen": result.choices[t_idx],
                    "reward": result.rewards[t_idx],
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parametrized smoke test: configuration matrix
# ---------------------------------------------------------------------------

# Cartesian product of fitting configuration flags.
_CONFIG_MATRIX = [
    pytest.param(
        "hgf_2level", False, False,
        id="2level-noLaplace-noTight",
    ),
    pytest.param(
        "hgf_2level", False, True,
        id="2level-noLaplace-tightOmega3",
    ),
    pytest.param(
        "hgf_3level", False, False,
        id="3level-noLaplace-noTight",
    ),
    pytest.param(
        "hgf_3level", False, True,
        id="3level-noLaplace-tightOmega3",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    "model_name, use_laplace_warmup, tight_omega3_prior",
    _CONFIG_MATRIX,
)
def test_fit_config_smoke(
    sim_df_small,
    model_name: str,
    use_laplace_warmup: bool,
    tight_omega3_prior: bool,
) -> None:
    """Smoke: fit_batch_hierarchical runs without exception for config combo.

    Parameters
    ----------
    sim_df_small : pandas.DataFrame
        Session-scoped synthetic cohort fixture.
    model_name : str
        HGF model variant (``"hgf_2level"`` or ``"hgf_3level"``).
    use_laplace_warmup : bool
        Whether to use Laplace-mode warmup initialization.
    tight_omega3_prior : bool
        Whether to apply tighter omega_3 prior (meaningful only for 3-level
        but must not crash on 2-level).
    """
    import arviz as az

    from prl_hgf.fitting.hierarchical import fit_batch_hierarchical

    result = fit_batch_hierarchical(
        sim_df_small,
        model_name=model_name,
        n_chains=2,
        n_draws=50,
        n_tune=50,
        target_accept=0.8,
        random_seed=42,
        sampler="blackjax",
        progressbar=False,
        use_laplace_warmup=use_laplace_warmup,
        tight_omega3_prior=tight_omega3_prior,
    )

    # Unpack if tuple (some code paths return (idata, extras_dict))
    if isinstance(result, tuple):
        idata = result[0]
    else:
        idata = result

    # --- Assertion 1: result is InferenceData ---
    assert isinstance(idata, az.InferenceData), (
        f"Expected az.InferenceData, got {type(idata).__name__}. "
        f"Config: model={model_name}, laplace={use_laplace_warmup}, "
        f"tight={tight_omega3_prior}"
    )

    # --- Assertion 2: posterior group exists ---
    assert hasattr(idata, "posterior"), (
        f"InferenceData missing 'posterior' group. "
        f"Available groups: {list(idata._groups)}. "
        f"Config: model={model_name}, laplace={use_laplace_warmup}, "
        f"tight={tight_omega3_prior}"
    )

    # --- Assertion 3: posterior is non-empty ---
    posterior = idata.posterior
    assert len(posterior.data_vars) > 0, (
        f"Posterior group has no data variables. "
        f"Config: model={model_name}, laplace={use_laplace_warmup}, "
        f"tight={tight_omega3_prior}"
    )
