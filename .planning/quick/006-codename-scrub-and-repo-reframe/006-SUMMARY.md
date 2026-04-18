# Quick Task 006: Codename Scrub + Repo Reframe

**Date:** 2026-04-18
**Scope:** A1 (working-tree scrub) + B1 (git history rewrite) + C1 (repo rename + reframe) per user's confidentiality directive
**Status:** Complete locally; two user-side follow-ups documented below

---

## What shipped

### A1 — Working-tree scrub (commit `10536fc`)

15 tracked files modified: 12 had heart2adapt references scrubbed via sed;
3 were reframed (pyproject.toml, README.md, CLAUDE.md).

Replacements applied:

- `HEART2ADAPT-sim` / `heart2adapt-sim` → `downstream sister-toolbox`
- `HEART2ADAPT` → `the consumer study`
- `Heart2Adapt` → `Consumer study`
- `heart2adapt` → `consumer-study`

Live code touched: one docstring in
`src/prl_hgf/analysis/export_trajectories.py:14`.

### B1 — Git history rewrite (force-pushed)

Installed `git-filter-repo` 2.47.0 via pip. Two passes:

1. `git filter-repo --replace-text .codename-replacements.txt --force` —
   rewrote file content in every historical commit (130 occurrences in
   `git log -p` diff lines → 2 remaining, both in commit messages).
2. `git filter-repo --replace-message .codename-replacements.txt --force` —
   rewrote commit messages (2 → 0).

**Final verification:** `git log --all -p | grep -c -i heart2adapt` = **0**.

Force-pushed to `origin/main`: `61f6c5a...10536fc main -> main (forced update)`.

**Backup tag** `pre-codename-scrub-backup-2026-04-18` on origin
(SHA `61f6c5a`) — rollback insurance. Safe to delete after ~24h once
cluster has been sync'd.

### C1 — Repo reframe (part of commit `10536fc`)

- `pyproject.toml`:
  - `name`: `prl-hgf` → `hgf-analysis`
  - `description`: psilocybin-specific → general-purpose HGF toolbox
- `README.md`: full rewrite
  - New title: "HGF Analysis Toolbox"
  - Supported-task table: pick_best_cue (psilocybin use case) alongside PAT-RL
  - Use-case section positions psilocybin study as one of several
- `CLAUDE.md`: Overview reframed
  - New title: "HGF Analysis Toolbox — AI Assistant Guidelines"
  - Both task structures documented (pick_best_cue + PAT-RL)
- Python package `prl_hgf` **unchanged** — already generic; `from prl_hgf
  import ...` still works identically

---

## User-side follow-ups (NOT automated — need you to run)

### 1. Cluster sync (CRITICAL — do this before next `sbatch`)

The force-push rewrites `origin/main`. The cluster's local clone has stale
history. On M3:

```bash
ssh m3
cd /fs04/fc37/adam/projects/psilocybin-prl-analyses    # or wherever
git fetch origin
git reset --hard origin/main
# verify no residual heart2adapt on the cluster side:
git grep -i heart2adapt || echo "clean"
```

Do NOT run `git pull` — it will try to merge the pre-rewrite local head
against the rewritten remote, which produces a mess. Use `reset --hard`.

If the cluster has any uncommitted local changes you want to preserve,
stash them BEFORE the reset (`git stash`) and re-apply after.

### 2. GitHub repo rename (your action, not mine — requires repo ownership)

Option A (web UI): https://github.com/adammanoogian/psilocybin-prl-analyses/settings
→ rename to `hgf-analysis`. GitHub sets up an automatic redirect from the
old URL, so existing clones continue to work.

Option B (CLI):
```bash
gh repo rename hgf-analysis --repo adammanoogian/psilocybin-prl-analyses
```

After rename, update local remote URL (both your machine and the cluster):
```bash
git remote set-url origin https://github.com/adammanoogian/hgf-analysis.git
```

### 3. Local folder rename (optional, breaks cluster-side path)

The local folder name `psilocybin_prl_analyses` is independent of the
GitHub repo name and just a filesystem choice. If you want to rename:

```bash
# Local machine (close editors/terminals in the folder first):
cd C:\Users\aman0087\Documents\Github
mv psilocybin_prl_analyses hgf_analysis

# Cluster (SLURM scripts may reference the old absolute path):
cd /fs04/fc37/adam/projects
mv psilocybin-prl-analyses hgf-analysis
# then grep SLURM scripts for any hardcoded path references:
grep -rn "psilocybin-prl-analyses" ~/projects/hgf-analysis/cluster/ 2>/dev/null
```

If you skip this step, the folder paths stay as-is; everything still works.

### 4. (Optional) delete the backup tag after 24h

Once the cluster is sync'd and the rewrite is confirmed stable:

```bash
git tag -d pre-codename-scrub-backup-2026-04-18          # local
git push origin --delete pre-codename-scrub-backup-2026-04-18  # remote
```

---

## Verification

- [x] `git grep -c -i heart2adapt` returns 0 on tracked content
- [x] `git log --all -p | grep -c -i heart2adapt` returns 0 (history scrubbed)
- [x] Force-push to origin succeeded (`61f6c5a...10536fc`)
- [x] Backup tag on origin at pre-rewrite commit (`61f6c5a`)
- [x] Working tree clean
- [x] pyproject.toml `name = "hgf-analysis"`
- [x] README.md + CLAUDE.md reframed as general HGF toolbox

---

## Known residuals (acceptable)

1. **Local folder path** still `psilocybin_prl_analyses` (user-side rename per above).
2. **GitHub repo URL** still `psilocybin-prl-analyses` (user-side rename per above).
3. **pip distribution name** changed `prl-hgf` → `hgf-analysis`. Downstream
   consumers that have `prl-hgf` pinned in their `pyproject.toml` will need
   to update the dep name (the import `from prl_hgf import ...` still works).
   Specifically `dcm_hgf_mixed_models/pyproject.toml` may need updating.
4. **Git reflog** on local machine still contains the pre-rewrite commits
   (`git reflog` shows them). This is local-only and expires after 90 days
   by default. Not visible to anyone cloning.

---

## Path forward

With scrub + reframe complete, Phase 20 planning can proceed from a clean
slate. The Phase 20 ROADMAP entry is already agnostic (updated earlier
in the session). Run:

```
/gsd:plan-phase 20
```

…to spawn the Phase 20 researcher, which will now read agnostic artifacts
and produce agnostic outputs.
