"""Integration test: shard_map chain dispatch produces distinct samples (P7).

Validates that each chain receives a unique PRNG key and produces
statistically different samples.  This prevents the Pitfall P7 scenario
where pmap/shard_map accidentally broadcasts the same RNG to all devices,
producing identical chains.

The test mocks 2 JAX CPU devices (via XLA_FLAGS) and runs 4 chains on a
trivial Normal posterior, asserting that chain samples differ byte-for-byte.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_chains_differ_across_devices() -> None:
    """Chains produce distinct samples when sharded across 2 CPU devices.

    Uses subprocess with XLA_FLAGS='--xla_force_host_platform_device_count=2'
    to simulate multi-device without requiring real GPUs.  Runs 4 chains
    across 2 devices (2 chains per device via vmap inside shard_map) and
    checks all 6 pairwise chain combinations.
    """
    # Script runs inside a subprocess with 2 mock CPU devices.
    #
    # shard_map semantics: with n_chains=4 and n_devices=2, each device
    # processes a shard of 2 chains (local_n = 2).  Inside shard_map the
    # function receives the local slice -- keys of shape (local_n, 2) and
    # states with a leading (local_n, ...) dimension.  We use jax.vmap over
    # the local batch to apply _sample_one_chain to each chain independently.
    #
    # P7 assertion: if shard_map broadcasts the same key to all devices,
    # chains on different devices would be identical even though chains on
    # the same device differ (vmap produces distinct outputs for different
    # key indices).  Checking cross-device chain equality catches P7.
    script = """
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

assert jax.device_count() == 2, f"Expected 2 devices, got {jax.device_count()}"

# Trivial log-posterior: standard Normal over 5 dimensions
def logp(x):
    return -0.5 * jnp.sum(x ** 2)

import blackjax

nuts = blackjax.nuts(logp, step_size=0.5, inverse_mass_matrix=jnp.ones(5))
init_state = nuts.init(jnp.zeros(5))

# 4 chains across 2 devices (2 chains per device)
n_chains = 4
n_devices = 2
n_draws = 50
sample_key = jax.random.PRNGKey(42)
# Each chain gets a unique key (P7 prevention)
chain_keys = jax.random.split(sample_key, n_chains)  # (4, 2)

# Stack initial state: (n_chains, *state_dims)
rep_state = jax.tree_util.tree_map(
    lambda x: jnp.broadcast_to(x, (n_chains, *x.shape)),
    init_state,
)

from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

mesh = Mesh(np.array(jax.devices()[:n_devices]), axis_names=("chains",))


def _sample_one_chain(rng_key, state):
    # Receives a scalar chain: rng_key shape (2,), state is un-batched.
    def _step(s, k):
        new_s, info = nuts.step(k, s)
        return new_s, new_s
    keys = jax.random.split(rng_key, n_draws)
    _, all_states = lax.scan(_step, state, keys)
    return all_states


def _shard_body(local_keys, local_states):
    # local_keys: (local_n, 2), local_states: (local_n, *dims)
    # Apply _sample_one_chain independently to each local chain via vmap.
    return jax.vmap(_sample_one_chain)(local_keys, local_states)


@jax.jit
def run_sharded(keys, states):
    # check_rep=False: NUTS uses lax.while_loop (tree expansion) which has
    # no replication rule in shard_map.  This flag disables the check;
    # outputs are still correctly sharded (P("chains")).
    return shard_map(
        _shard_body,
        mesh=mesh,
        in_specs=(P("chains"), P("chains")),
        out_specs=P("chains"),
        check_rep=False,
    )(keys, states)


all_states = run_sharded(chain_keys, rep_state)
samples = np.asarray(all_states.position)  # (n_chains, n_draws, 5)

assert samples.shape == (n_chains, n_draws, 5), (
    f"Expected shape ({n_chains}, {n_draws}, 5), got {samples.shape}"
)

# Assertion 1: all 6 pairwise combinations must differ byte-for-byte
# Chains on different devices differ because they get different keys.
# Chains on the same device differ because vmap uses different key indices.
for i in range(n_chains):
    for j in range(i + 1, n_chains):
        assert not np.array_equal(samples[i], samples[j]), (
            f"P7 FAILURE: chain {i} and chain {j} are byte-for-byte identical! "
            "shard_map is broadcasting the same RNG key."
        )

# Assertion 2: pairwise correlation must be low (independent chains)
for i in range(n_chains):
    for j in range(i + 1, n_chains):
        corr = np.corrcoef(samples[i].ravel(), samples[j].ravel())[0, 1]
        assert abs(corr) < 0.5, (
            f"Chains {i} and {j} suspiciously correlated: r={corr:.3f}. "
            "Expected near-zero correlation for independent NUTS chains."
        )

print("P7 chain-divergence test PASSED (4 chains, 2 devices)")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=2"},
    )
    assert result.returncode == 0, (
        f"Chain divergence test failed!\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "P7 chain-divergence test PASSED (4 chains, 2 devices)" in result.stdout, (
        result.stdout
    )
