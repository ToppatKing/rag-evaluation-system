# Git Snapshot Guide — Preserving the Baseline

This document explains how to create a permanent, runnable snapshot of your
current codebase (the "before HyDE" version) and how to switch between it and
the new version at any time — for demos, for showing professors the before/after
diff, or for rolling back if something breaks.

---

## Step 0 — Make sure everything is committed

Run this in your repo root before touching anything:

```bash
git status
git add -A
git commit -m "chore: clean state before HyDE addition"
```

---

## Step 1 — Tag the baseline (the snapshot)

A **tag** is a permanent, named pointer to a specific commit. It never moves,
even as you add new commits later.  This becomes your "before" screenshot that
you can show at the thesis meeting.

```bash
# Annotated tag — includes a message, timestamp, and author
git tag -a v1.0-baseline \
        -m "Baseline: Dense + MMR retrieval only, no HyDE, no CUAD"

# Push the tag to GitHub so it's also there as a permanent record
git push origin v1.0-baseline
```

On GitHub it will appear under **Releases → Tags** and you (and your professors)
can browse it at:  `https://github.com/ToppatKing/rag-evaluation-system/tree/v1.0-baseline`

---

## Step 2 — Create a feature branch for your changes

**Never commit directly to `master` during thesis development.**  A branch lets
you keep `master` == `v1.0-baseline` exactly, so the baseline is always one
`git checkout` away.

```bash
git checkout -b feature/hyde-cuad
```

---

## Step 3 — Copy in the new files

Copy each file from this `rag_thesis/` folder to the right place in your repo:

| This file                         | Goes to                                          |
|-----------------------------------|--------------------------------------------------|
| `retriever.py`                    | `src/rag_eval/retrieval/retriever.py`            |
| `generator_patch.py`              | Read the instructions inside; patch your existing `generator.py` |
| `config.yaml`                     | `config/config.yaml`                             |
| `setup_cuad.py`                   | `scripts/setup_cuad.py`                          |
| `run_ablation.py`                 | `scripts/run_ablation.py`                        |

Then install the one new dependency:

```bash
pip install datasets        # HuggingFace datasets — needed by setup_cuad.py
```

And add it to `pyproject.toml`:

```toml
[project.optional-dependencies]
cuad = ["datasets>=2.14"]
```

---

## Step 4 — Commit the new version

```bash
git add -A
git commit -m "feat: add HyDE retriever + CUAD ingestion + ablation runner"

# Also tag this version
git tag -a v2.0-hyde -m "v2.0: HyDE + Dense + MMR ablation on CUAD"
git push origin feature/hyde-cuad --tags
```

---

## How to switch between versions

### Show the baseline (old code) — for a demo or prof meeting

```bash
# Switch to the baseline tag (read-only, "detached HEAD")
git checkout v1.0-baseline

# Run it exactly as it was
python scripts/run_demo.py --query "What is the governing law?"

# Come back to your working branch when done
git checkout feature/hyde-cuad
```

### Show the diff of every change you made

```bash
# Compare baseline tag to your current branch — shows ALL changes
git diff v1.0-baseline feature/hyde-cuad

# Show only which files changed
git diff --name-only v1.0-baseline feature/hyde-cuad

# Show the diff for just the retriever file
git diff v1.0-baseline feature/hyde-cuad -- src/rag_eval/retrieval/retriever.py
```

### Create a GitHub Pull Request to show the before/after

On GitHub, open a PR from `feature/hyde-cuad` → `master`.  The PR diff view
shows every added/removed line with colour coding — perfect for a thesis
appendix screenshot or for walking a professor through "here is exactly what
changed."

---

## Showing the two versions side-by-side to professors

Option A — Two terminal windows:
```bash
# Terminal 1 (baseline)
git worktree add /tmp/rag-baseline v1.0-baseline
cd /tmp/rag-baseline
python scripts/run_demo.py --query "What is the governing law?"

# Terminal 2 (new version, current directory)
python scripts/run_demo.py --query "What is the governing law?"
```

Option B — Run the ablation script (shows all three modes in one table):
```bash
python scripts/run_ablation.py \
    --config config/config.yaml \
    --dataset data/cuad_eval.json \
    --output results/cuad_ablation/
```
The output `ablation_report.txt` is the exact table to put in your thesis.

---

## Summary of the tag/branch strategy

```
master ────────────────────────────────────────────────────────►
        │
        │ (tag: v1.0-baseline — permanent snapshot, never changes)
        │
        └── feature/hyde-cuad ─── commit ─── commit ─── ► HEAD
                                    │
                                    │ (tag: v2.0-hyde)
```

The key rule: **tags never move**.  `v1.0-baseline` will always point to
exactly the code as it was before you changed anything, no matter how many
commits you add later.
