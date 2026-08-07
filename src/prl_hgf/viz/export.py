"""HGF viewer HTML export.

Phase 23 public API: ``render_viewer_html`` -- a single function that
turns a :class:`prl_hgf.viz.schema.NetworkSpec` into a self-contained
browser-openable HTML string.

Architecture (three mechanics, each load-bearing):

1. **String-surgery JSON injection** (Strategy B from research §2):
   the template ``reports/figures/hgf_viewer.html`` ships with valid default
   JSON payloads baked into ``<script type="application/json"
   id="viz-X">`` marker tags. ``_inject_markers`` regex-replaces the
   content between the marker open/close tags -- preserving the
   "browser-openable without Python" property of the raw template.

2. **Jinja2 CDN block** (research §3 + §10.10): the template's three
   CDN ``<script src=...>`` tags are wrapped in ``{% if react_inlined
   %}...{% else %}...{% endif %}``. In ``offline_mode=False``, the
   CDN assets are fetched via ``urllib.request`` and inlined.

   **Custom variable delimiters** (deviation Rule 1, plan 23-03 task 2):
   the React/Babel JSX inside the template uses ``{{...}}`` heavily for
   inline-style and JS-expression embedding (e.g. ``style={{cursor:
   "pointer"}}``). Default Jinja2 ``variable_start_string="{{"`` would
   try to evaluate every JSX expression as a Jinja2 variable and fail
   with ``TemplateSyntaxError``. Both ``Environment`` instances in this
   module set ``variable_start_string="<<"`` /
   ``variable_end_string=">>"`` (neither token appears anywhere in the
   template), and ``reports/figures/hgf_viewer.html`` uses ``<< react_inlined
   | safe >>`` for the three CDN inlining lines accordingly. The
   ``{% %}`` statement delimiters keep their default form (no JSX
   collision -- the template uses ``{%`` only on the three Jinja2 if/
   else/endif lines, all HTML-comment-wrapped per plan 23-02).

3. **MDN-canonical JSON-in-HTML escape** (research §4, finding #1): the
   ``_json_for_html`` helper pairs ``json.dumps(ensure_ascii=True)``
   with ``.replace("</", "<\\/")`` -- both defenses are required
   because ``ensure_ascii=True`` alone does NOT escape the
   ``</script>`` tag-close sequence (empirically verified in ds_env,
   2026-04-24).

Parallel-stack invariant: this module imports from stdlib,
``jinja2``, ``prl_hgf.viz.schema``, and top-level ``config`` (for
``PROJECT_ROOT`` -- CLAUDE.md mandate). No imports from
``prl_hgf.{env, models, fitting, analysis, gui, power, simulation}``.
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment

# NOTE: ``config`` is the top-level path-constants module (repo-root
# ``config.py``, NOT a prl_hgf subpackage). CLAUDE.md mandates config.py
# as the source of Path constants; config.py is not in the
# parallel-stack forbidden list (prl_hgf.{env,models,fitting,analysis,
# gui,power,simulation}).
from prl_hgf._paths import PROJECT_ROOT
from prl_hgf.viz.schema import NetworkSpec

__all__ = ["render_viewer_html"]


def _json_for_html(obj: Any) -> str:
    """Serialize ``obj`` as JSON safe to embed inside ``<script type="application/json">``.

    Two defenses combined (research §4, finding #1):

    - ``ensure_ascii=True`` escapes control chars, U+2028/U+2029, and
      non-ASCII codepoints.
    - ``.replace("</", "<\\/")`` escapes the HTML tag-close sequence.
      This is REQUIRED because ``ensure_ascii=True`` does NOT escape
      ``</`` on its own (the slash is ASCII). Omitting this defense
      lets a user-authored string like ``"<script>alert(1)</script>"``
      break out of the ``<script>`` container and trigger XSS.

    Inside a JSON string literal, ``<\\/script>`` is a valid-JSON
    escape that deserializes back to ``</script>`` via ``JSON.parse``.

    Parameters
    ----------
    obj
        Any JSON-serializable Python object (typically a dict produced
        by ``dataclasses.asdict(network_spec)``).

    Returns
    -------
    str
        ASCII-only JSON string with ``</`` sequences neutralized to
        ``<\\/``. Safe for embedding inside ``<script
        type="application/json">`` tags.
    """
    return json.dumps(obj, ensure_ascii=True).replace("</", "<\\/")


_MARKER_RE = re.compile(
    r'(<script type="application/json" id="viz-([a-z\-]+)">)(.*?)(</script>)',
    re.DOTALL,
)


def _inject_markers(template: str, payload: dict[str, str]) -> str:
    """Replace content between ``<script type="application/json" id="viz-X">`` markers.

    Parameters
    ----------
    template
        The raw template string (loaded from ``reports/figures/hgf_viewer.html``).
    payload
        Mapping of short marker name (e.g. ``"network-spec"``, NOT
        ``"viz-network-spec"``) to the already-serialized JSON string
        via :func:`_json_for_html`.

    Returns
    -------
    str
        The template with every matching marker's content replaced.
        Unmatched markers in the template remain with their default body
        (the valid JSON blob baked in at plan 23-02).

    Raises
    ------
    ValueError
        If any key in ``payload`` does not correspond to a marker in the
        template. Message format: ``"injection marker 'viz-{name}' not
        found in template"`` where ``{name}`` is the first missing name
        in sorted order (deterministic).
    """
    replaced_names: set[str] = set()

    def _replace(m: re.Match[str]) -> str:
        name = m.group(2)
        if name in payload:
            replaced_names.add(name)
            return f"{m.group(1)}\n{payload[name]}\n{m.group(4)}"
        return m.group(0)

    out = _MARKER_RE.sub(_replace, template)
    missing = set(payload) - replaced_names
    if missing:
        first_missing = sorted(missing)[0]
        raise ValueError(
            f"injection marker 'viz-{first_missing}' not found in template"
        )
    return out


_CDN_SOURCES: dict[str, str] = {
    "react": "https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js",
    "react_dom": "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.production.min.js",
    "babel": "https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js",
}


def _fetch_cdn_asset(url: str, timeout: float = 30.0) -> str:
    """Fetch a CDN JS asset as a UTF-8 string.

    Uses ``ssl.create_default_context()`` -- Windows-safe per research §3
    (certifi-bundled via conda-forge ds_env). Do NOT use
    ``ssl.CERT_NONE`` -- the correct fix for any Windows cert issue is
    ``conda update certifi`` or setting ``SSL_CERT_FILE``.

    Parameters
    ----------
    url
        Fully-qualified HTTPS URL (only cdnjs.cloudflare.com and
        unpkg.com are expected; both peer at <100 ms from ds_env).
    timeout
        Connection + read timeout in seconds. Default 30 s is generous;
        Babel at 2.8 MiB may take 1-2 s on average.

    Returns
    -------
    str
        UTF-8 decoded body.
    """
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:  # noqa: S310
        raw: bytes = resp.read()
    return raw.decode("utf-8")


def _make_jinja_env() -> Environment:
    """Build the Jinja2 ``Environment`` used by both render helpers.

    Uses ``BaseLoader()`` (template is passed in as a string, never
    loaded from disk by Jinja2 -- :func:`render_viewer_html` does the
    file read directly) and ``autoescape=False`` because the inlined JS
    source MUST pass through unescaped (research §10.10).

    Sets custom variable delimiters ``<<``/``>>`` to avoid colliding
    with the template's heavy use of JSX ``{{...}}`` patterns (deviation
    Rule 1, plan 23-03 task 2). Statement delimiters ``{% %}`` keep
    their default form.

    Returns
    -------
    Environment
        Configured Jinja2 environment ready for ``env.from_string``.
    """
    return Environment(  # noqa: S701  -- autoescape disabled by design (see docstring)
        loader=BaseLoader(),
        autoescape=False,
        variable_start_string="<<",
        variable_end_string=">>",
    )


def _inline_cdn_assets(
    template: str,
    *,
    fetch_timeout: float = 30.0,
) -> str:
    """Inline React + React-DOM + Babel CDN assets into the template.

    The template is expected to contain a ``{% if react_inlined and
    react_dom_inlined and babel_inlined %}...{% else %}<script
    src="https://cdnjs...">...{% endif %}`` block (landed in plan 23-02)
    with three ``<< react_inlined | safe >>``-style insertion points
    inside the if-branch. When this helper is called, the ``{% if %}``
    branch becomes live and the three CDN asset bodies are substituted
    in.

    Parameters
    ----------
    template
        Raw template string with the Jinja2 if/else block.
    fetch_timeout
        Passed to :func:`_fetch_cdn_asset` for each of the three assets.

    Returns
    -------
    str
        Template with ``<< react_inlined >>``, ``<< react_dom_inlined >>``,
        ``<< babel_inlined >>`` substituted with UTF-8 decoded asset
        bodies.
    """
    env = _make_jinja_env()
    jinja_template = env.from_string(template)
    return jinja_template.render(
        react_inlined=_fetch_cdn_asset(_CDN_SOURCES["react"], fetch_timeout),
        react_dom_inlined=_fetch_cdn_asset(_CDN_SOURCES["react_dom"], fetch_timeout),
        babel_inlined=_fetch_cdn_asset(_CDN_SOURCES["babel"], fetch_timeout),
    )


def _render_offline_template(template: str) -> str:
    """Render the template without CDN inlining -- Jinja2 else-branch path.

    When ``offline_mode=True`` (default), the output HTML keeps the
    CDN ``<script src=...>`` tags; user's browser fetches React + Babel
    at file-open time. No network calls at render time.

    Passing ``None`` for the three Jinja2 variables makes the template's
    ``{% if react_inlined and react_dom_inlined and babel_inlined %}``
    evaluate falsy, so the ``{% else %}`` branch (CDN ``<script src>``
    tags) renders.

    Parameters
    ----------
    template
        Raw template string with the Jinja2 if/else block.

    Returns
    -------
    str
        Template with the CDN-tag else-branch rendered live; no network
        I/O occurs.
    """
    env = _make_jinja_env()
    jinja_template = env.from_string(template)
    return jinja_template.render(
        react_inlined=None,
        react_dom_inlined=None,
        babel_inlined=None,
    )


# Default template path, derived from the sanctioned top-level
# ``config.PROJECT_ROOT`` (CLAUDE.md mandate -- "Config via config.py
# Path constants -- no scattered hardcoded paths"). This replaces the
# previous draft's ``Path(__file__).resolve().parents[3]`` hand-counted
# index, which silently broke under any packaging move.
_DEFAULT_TEMPLATE_PATH: Path = PROJECT_ROOT / "reports" / "figures" / "hgf_viewer.html"

# Import-time existence assertion -- a misconfigured/moved template
# fails loudly with a specific FileNotFoundError at
# ``import prl_hgf.viz.export`` rather than deep inside a render call.
if not _DEFAULT_TEMPLATE_PATH.exists():
    raise FileNotFoundError(
        f"HGF viewer template missing at import time: {_DEFAULT_TEMPLATE_PATH}. "
        f"Expected: config.PROJECT_ROOT / 'reports' / 'figures' / 'hgf_viewer.html'. "
        f"Has the template been moved, or has config.PROJECT_ROOT shifted? "
        f"(Plan 23-02 lands this file; check git log -- reports/figures/hgf_viewer.html)"
    )


def render_viewer_html(
    spec: NetworkSpec,
    *,
    offline_mode: bool = True,
    template_path: Path | None = None,
    fetch_timeout: float = 30.0,
) -> str:
    """Render a NetworkSpec to a self-contained HGF viewer HTML string.

    Parameters
    ----------
    spec
        A :class:`prl_hgf.viz.schema.NetworkSpec` -- typically produced
        by :func:`prl_hgf.viz.schema.build_network_spec`. Pre-fit and
        post-fit specs are both supported; post-fit handling is the
        Phase 24 extension (POST-02 populates viz-posterior-summary).
    offline_mode
        If True (default), the returned HTML contains ``<script
        src="https://cdnjs...">`` tags for React/React-DOM/Babel -- NO
        network I/O at render time; user's browser fetches from CDN at
        file-open time. Use this for tests and for the "just open the
        file" workflow. Semantic note: "offline" here means "render
        performs no network I/O", NOT "opens without internet".

        If False, React/React-DOM/Babel are fetched via
        ``urllib.request`` and inlined in the returned HTML. Result is
        a single ~2.9 MiB self-contained HTML file that opens with no
        network dependency. Use this for the "distribute one file"
        workflow.
    template_path
        Override the default ``reports/figures/hgf_viewer.html`` location. For
        testing.
    fetch_timeout
        Passed through to CDN fetch helpers when ``offline_mode=False``.

    Returns
    -------
    str
        A complete HTML document as a string. ``bs4.BeautifulSoup(html,
        "html.parser")`` parses it without errors. Both
        ``viz-network-spec`` and ``viz-posterior-summary`` script tags
        are present.

    Raises
    ------
    ValueError
        From :func:`_inject_markers` if the template is missing a
        required injection marker (EXPORT-04 contract).
    FileNotFoundError
        If ``template_path`` (or the default) does not exist at call
        time (the module-level existence check catches the default case
        at import time).
    """
    path = template_path if template_path is not None else _DEFAULT_TEMPLATE_PATH
    template = path.read_text(encoding="utf-8")

    # Step 1: String-surgery injection of JSON payloads (Strategy B).
    # Pre-fit: viz-posterior-summary gets empty body (intentional signal;
    # Phase 24 POST-02 populates when idata is supplied).
    spec_dict = asdict(spec)
    network_spec_json = _json_for_html(spec_dict)
    posterior_summary_json = ""  # pre-fit sentinel; empty body

    template_with_data = _inject_markers(
        template,
        {
            "network-spec": network_spec_json,
            "posterior-summary": posterior_summary_json,
        },
    )

    # Step 2: Jinja2 render for the {% if react_inlined %} block.
    if offline_mode:
        return _render_offline_template(template_with_data)
    return _inline_cdn_assets(template_with_data, fetch_timeout=fetch_timeout)
