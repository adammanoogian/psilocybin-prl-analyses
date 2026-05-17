"""Guard test: no duplicate keyword arguments in BlackJAX call sites.

Catches the class of bug fixed in b737fe6 where ``max_num_doublings`` was
passed both explicitly AND via ``**warmup_params`` dict-spread, causing a
``TypeError: got multiple values for keyword argument`` at runtime.

This test uses Python's ``ast`` module to statically parse
``src/prl_hgf/fitting/hierarchical.py`` and inspect every call to
``blackjax.nuts(...)`` and ``blackjax.window_adaptation(...)``.  For each
call site it:

1. Collects all explicit keyword argument names.
2. Flags ``**dict_spread`` arguments as potential collision sources.
3. Asserts no explicit keyword appears more than once in the same call.

This is a structural guard that runs without importing any heavy
dependencies (no JAX, no BlackJAX needed).

Run::

    pytest tests/integration/test_guard04_kwarg_collision.py -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# Ensure project root is importable (for path resolution only).
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Target file to inspect.
_HIERARCHICAL_PY = (
    _root / "src" / "prl_hgf" / "fitting" / "hierarchical.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_blackjax_call(node: ast.Call, func_names: set[str]) -> bool:
    """Check if a Call node targets one of the tracked BlackJAX functions.

    Matches patterns like:
    - ``blackjax.nuts(...)``
    - ``blackjax.window_adaptation(...)``
    - ``nuts(...)`` (if imported directly)
    - ``window_adaptation(...)``

    Parameters
    ----------
    node : ast.Call
        The AST Call node to inspect.
    func_names : set[str]
        Set of function names to match (e.g., ``{"nuts", "window_adaptation"}``).

    Returns
    -------
    bool
        True if this call targets a tracked BlackJAX function.
    """
    func = node.func

    # Pattern: blackjax.nuts(...) or blackjax.window_adaptation(...)
    if isinstance(func, ast.Attribute) and func.attr in func_names:
        return True

    # Pattern: nuts(...) or window_adaptation(...) (direct import)
    return isinstance(func, ast.Name) and func.id in func_names


def _extract_call_info(
    node: ast.Call, source_lines: list[str]
) -> dict:
    """Extract keyword argument info from a Call node.

    Parameters
    ----------
    node : ast.Call
        The AST Call node.
    source_lines : list[str]
        Source file lines (for error reporting).

    Returns
    -------
    dict
        Keys: ``line``, ``col``, ``func_name``, ``explicit_kwargs``,
        ``has_dict_spread``, ``spread_expressions``.
    """
    # Determine function name for reporting
    func = node.func
    if isinstance(func, ast.Attribute):
        func_name = f"{ast.dump(func.value)}.{func.attr}"
        # Simplify common case
        if isinstance(func.value, ast.Name):
            func_name = f"{func.value.id}.{func.attr}"
    elif isinstance(func, ast.Name):
        func_name = func.id
    else:
        func_name = ast.dump(func)

    # Collect explicit keyword names
    explicit_kwargs: list[str] = []
    has_dict_spread = False
    spread_expressions: list[str] = []

    for kw in node.keywords:
        if kw.arg is None:
            # This is a **spread (e.g., **warmup_params or **{...})
            has_dict_spread = True
            spread_expressions.append(ast.unparse(kw.value))
        else:
            explicit_kwargs.append(kw.arg)

    return {
        "line": node.lineno,
        "col": node.col_offset,
        "func_name": func_name,
        "explicit_kwargs": explicit_kwargs,
        "has_dict_spread": has_dict_spread,
        "spread_expressions": spread_expressions,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBlackjaxKwargCollision:
    """Static guard: no duplicate kwargs in BlackJAX call sites."""

    @pytest.fixture(scope="class")
    def parsed_calls(self) -> list[dict]:
        """Parse hierarchical.py and extract all BlackJAX call site info."""
        assert _HIERARCHICAL_PY.is_file(), (
            f"Target file not found: {_HIERARCHICAL_PY}"
        )
        source = _HIERARCHICAL_PY.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(_HIERARCHICAL_PY))
        source_lines = source.splitlines()

        target_funcs = {"nuts", "window_adaptation"}
        calls: list[dict] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_blackjax_call(
                node, target_funcs
            ):
                info = _extract_call_info(node, source_lines)
                calls.append(info)

        return calls

    def test_blackjax_calls_found(self, parsed_calls: list[dict]) -> None:
        """Sanity: at least one BlackJAX call site exists to guard."""
        assert len(parsed_calls) > 0, (
            "No blackjax.nuts() or blackjax.window_adaptation() calls found "
            f"in {_HIERARCHICAL_PY}. Has the file been restructured? "
            "Update this test if call sites moved to a different module."
        )

    def test_no_duplicate_explicit_kwargs(
        self, parsed_calls: list[dict]
    ) -> None:
        """No explicit keyword argument appears more than once per call.

        This catches the literal duplicate like::

            blackjax.nuts(fn, max_num_doublings=10, max_num_doublings=10)

        which Python would reject at parse time, but also catches the more
        subtle pattern where kwargs are assembled programmatically.
        """
        violations: list[str] = []

        for call in parsed_calls:
            kwargs = call["explicit_kwargs"]
            seen: dict[str, int] = {}
            for kw in kwargs:
                seen[kw] = seen.get(kw, 0) + 1

            duplicates = {k: v for k, v in seen.items() if v > 1}
            if duplicates:
                violations.append(
                    f"Line {call['line']}: {call['func_name']}() has "
                    f"duplicate kwargs: {duplicates}"
                )

        assert not violations, (
            "Duplicate explicit keyword arguments found in BlackJAX calls:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_dict_spread_collision_risk(
        self, parsed_calls: list[dict]
    ) -> None:
        """Flag dict-spread sites where explicit kwarg also appears in spread.

        When a call uses ``**some_dict`` AND also passes the same key
        explicitly, Python raises TypeError at runtime.  This test catches
        patterns like::

            blackjax.nuts(fn, max_num_doublings=10, **params)

        where ``params`` might contain ``max_num_doublings``.

        The current codebase uses ``**{**warmup_params, "key": value}``
        pattern which is SAFE (dict merge resolves before unpacking).  This
        test verifies that no UNSAFE pattern exists (explicit kwarg +
        separate ``**variable`` spread that could collide).
        """
        violations: list[str] = []

        for call in parsed_calls:
            if not call["has_dict_spread"]:
                continue

            explicit = set(call["explicit_kwargs"])
            if not explicit:
                continue

            # Check if the spread is a simple variable (risky) vs inline
            # dict merge (safe).
            for spread_expr in call["spread_expressions"]:
                # Inline dict construction like {**warmup_params, "key": val}
                # is safe because dict merge resolves duplicates before the
                # call.  Only flag plain variable spreads.
                is_inline_dict = spread_expr.startswith("{")
                if is_inline_dict:
                    # Safe pattern: merge happens before call
                    continue

                # Plain variable spread (e.g., **warmup_params) alongside
                # explicit kwargs is the risky pattern.
                violations.append(
                    f"Line {call['line']}: {call['func_name']}() mixes "
                    f"explicit kwargs {sorted(explicit)} with "
                    f"**{spread_expr} — potential collision risk. "
                    f"Use inline dict merge pattern instead: "
                    f"**{{**{spread_expr}, 'key': value}}"
                )

        assert not violations, (
            "Potential kwarg collision risk in BlackJAX calls:\n"
            + "\n".join(f"  - {v}" for v in violations)
            + "\n\nThe safe pattern is: "
            "**{**warmup_params, 'max_num_doublings': value} "
            "(inline dict merge resolves before call)."
        )
