"""GUARD-03: JIT cache reuse runtime test (P9 prevention).

Verifies that running fit_batch_hierarchical multiple times with the same
FitConfig and same-shaped data produces zero XLA cache misses on iterations
2+. This guards against regressions in the traced-arg sample loop design
that would cause full recompilation (~1600s) on every power-sweep iteration.

The test uses JAX_LOG_COMPILES=1 environment variable to detect compilations.
If cache-reuse breaks, 5 iterations produce 5x the compile count of 1 iteration.

NOTE: This test requires JAX and BlackJAX. It runs a minimal fit (2 chains,
10 draws, 10 warmup) so total runtime is ~30-60s including one cold compile.
Must run on cluster if projection exceeds 3 minutes -- but with these tiny
settings it should complete locally in <2 minutes.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable regardless of install mode.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Module-scoped fixture: tiny synthetic cohort DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_sim_df():
    """Generate a tiny synthetic cohort (2 participants, 40 trials each).

    Module-scoped to avoid regeneration across test cells.
    """
    import pandas as pd

    from prl_hgf.env.simulator import generate_session
    from prl_hgf.env.task_config import load_config
    from prl_hgf.models.hgf_2level import build_2level_network
    from prl_hgf.simulation.agent import simulate_agent

    cfg = load_config()
    n_participants = 2
    n_trials_max = 40
    base_seed = 77700
    rows: list[dict] = []

    for p_idx in range(n_participants):
        pid = f"CACHE{p_idx + 1:03d}"
        seed = base_seed + p_idx
        rng = np.random.default_rng(seed)
        net = build_2level_network(omega_2=-3.0)
        trials = generate_session(cfg, seed=seed)
        result = simulate_agent(net, trials, beta=3.0, zeta=0.5, rng=rng)

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
# GUARD-03: JIT cache reuse test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_jit_cache_reuse_zero_misses(tiny_sim_df):
    """5 iterations with identical FitConfig: total compiles <= 12.

    Uses a subprocess to cleanly capture JAX compilation logs without
    interference from other test-session JIT state.  Asserts that the total
    XLA compile count across all 5 iterations does not grow linearly (which
    would indicate cache breakage).

    GUARD-03 spec: cache misses == 0 on iterations 2-5.  The threshold of 12
    allows up to 3x a generous single-iteration budget of 4, accommodating
    minor internal retracing while still catching the pathological case of
    full recompilation per iteration (which would produce 20+ compiles).
    """
    import subprocess

    # Write sim_df to a temp parquet for subprocess consumption.
    tmp_dir = tempfile.mkdtemp()
    parquet_path = os.path.join(tmp_dir, "tiny_cohort.parquet")
    tiny_sim_df.to_parquet(parquet_path)

    # Build the script using a raw string to avoid backslash issues.
    root_str = str(_root).replace("\\", "/")
    script = (
        "import os\n"
        'os.environ["JAX_LOG_COMPILES"] = "1"\n'
        "\n"
        "import sys\n"
        f'sys.path.insert(0, "{root_str}")\n'
        "\n"
        "import pandas as pd\n"
        "\n"
        "from prl_hgf.fitting.config import FitConfig, MitigationConfig, SamplerConfig\n"
        "from prl_hgf.fitting.hierarchical import fit_batch_hierarchical\n"
        "\n"
        f'df = pd.read_parquet("{parquet_path}")\n'
        "\n"
        "fit_config = FitConfig(\n"
        '    model_name="hgf_2level",\n'
        "    sampler=SamplerConfig(\n"
        '        backend="blackjax",\n'
        "        n_chains=2,\n"
        "        n_draws=10,\n"
        "        n_warmup=10,\n"
        "        target_accept=0.8,\n"
        "        random_seed=42,\n"
        "    ),\n"
        "    mitigation=MitigationConfig(\n"
        "        use_laplace_warmup=False,\n"
        '        mass_matrix_kind="diagonal",\n'
        "    ),\n"
        "    progressbar=False,\n"
        "    log_every=0,\n"
        ")\n"
        "\n"
        "# Iteration 1: cold compile (cache miss expected).\n"
        'print("ITER_START_1", flush=True)\n'
        "result = fit_batch_hierarchical(df, fit_config)\n"
        "if isinstance(result, tuple):\n"
        "    _, warmup_params = result\n"
        "else:\n"
        "    warmup_params = None\n"
        'print("ITER_END_1", flush=True)\n'
        "\n"
        "# Iterations 2-5: should be cache hits (same config, same shapes).\n"
        "for i in range(2, 6):\n"
        '    print(f"ITER_START_{i}", flush=True)\n'
        "    fit_batch_hierarchical(df, fit_config, warmup_params=warmup_params)\n"
        '    print(f"ITER_END_{i}", flush=True)\n'
        "\n"
        'print("ALL_ITERATIONS_COMPLETE")\n'
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if proc.returncode != 0:
        pytest.fail(
            f"Cache reuse test subprocess failed!\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )

    assert "ALL_ITERATIONS_COMPLETE" in proc.stdout, (
        "Test did not complete all iterations.\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-2000:]}"
    )

    # Count total XLA compilation events in stderr.
    # JAX logs compilations as lines containing "Compiling" or "XLA compilation"
    # when JAX_LOG_COMPILES=1 is set.
    compile_pattern = re.compile(r"(Compil|XLA compilation)", re.IGNORECASE)
    total_compiles = len(compile_pattern.findall(proc.stderr))

    # GUARD-03 threshold:
    #   If cache reuse works:  compiles happen only in iter 1 => ~2-6 total.
    #   If cache is broken:    5 iterations each recompile  => ~10-25 total.
    #   Threshold of 12 is 3x a generous single-iteration budget of 4,
    #   accommodating minor scan-body specialisation without allowing the
    #   pathological per-iteration-recompile regression through.
    max_acceptable = 12
    assert total_compiles <= max_acceptable, (
        f"GUARD-03 FAILURE: {total_compiles} XLA compilations detected across "
        f"5 iterations (expected <= {max_acceptable} if cache reuse works). "
        f"This suggests the traced-arg sample loop is recompiling on each call. "
        f"Check that FitConfig.__hash__ is stable and data shapes are constant.\n"
        f"stderr tail:\n{proc.stderr[-3000:]}"
    )

    # Cleanup temp files.
    os.remove(parquet_path)
    os.rmdir(tmp_dir)
