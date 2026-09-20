# Handoff: get `scripts/eval_naksha_seed.py` running

## Status as of 2026-09-19

**Confirmed facts (from direct file reads over the mount, reliable):**
- `smriti-core/` contains only `Cargo.toml` — no stray `pyproject.toml` in that directory.
  This confirms the assumption behind the fix below: once you `cd smriti-core` before running
  `maturin develop`, there's nothing else in that directory for maturin to mistakenly pick up.
- The Naksha seed JSON files (`../naksha/naksha/seed/data/*.json`) are unchanged since Aug 19 —
  same 5 files, same timestamps. The eval script's data source is stable.

**Not reliable — do not trust environment info relayed from outside your own shell:** any
toolchain version info (Python/pip/maturin/cargo/rustc versions) gathered via a bridged tool
from outside your actual terminal session is NOT trustworthy for this repo — there's a real
Linux sandbox in the loop that is architecturally separate from the macOS `.venv` this project
actually needs, and it produced misleading version numbers (a different Python, no maturin
installed at all) when queried that way. Check these yourself, directly, as your first step:
```
which python pip maturin cargo rustc
python --version
pip --version
maturin --version
cargo --version
```

Re-checked directly: both imports still fail in `smriti/.venv`.
```
python -c "import smriti_core"   # ModuleNotFoundError: No module named 'smriti_core'
python -c "from smriti.embeddings.mrl_pipeline import MRLEmbeddingPipeline"  # ModuleNotFoundError: No module named 'smriti'
```
Some doc/README work has landed since this was first diagnosed (see `git log`: "docs: document
Naksha adapter context and link in READMEs", "docs: add sub-package READMEs for smriti-core
(Rust) and src/smriti (Python)"), and a `uv.lock` now exists untracked in the working tree — but
the actual build/install issue below does not look resolved. This file itself was deleted at
some point after the first handoff (not committed, not present on disk) — recreating it now with
the same diagnosis plus this status update, since the underlying problem is unchanged.

## Goal
Run `scripts/eval_naksha_seed.py --fallback-embeddings` successfully in `smriti/.venv`.
It's a correctness/plumbing spike, not the recall/latency benchmark — it loads real Naksha
seed data (`../naksha/naksha/seed/data/*.json`, no SurrealDB/Docker needed), embeds it,
builds an HAM-P tree, and runs a few sanity checks (self-recall, domain purity, an
edge-discovery distance check, a resolution-dial sweep). The script already exists at
`scripts/eval_naksha_seed.py` (untracked in git — `git status` shows it as `??`, worth
committing once it runs) — don't rewrite it, just get the environment to a state where it
imports cleanly.

## The failure mode (confirmed, reproduced repeatedly across two sessions now)
Two things need to both be importable from the SAME venv (`smriti/.venv`, Python 3.12, arm64 macOS):
- `smriti_core` — the compiled Rust/PyO3 native extension (from `smriti-core/Cargo.toml`)
- `smriti` — the pure-Python package (`src/smriti/...`, installed via `pip install -e .` off
  the repo-root `pyproject.toml`, which declares `name = "smriti"`)

These are two separate installs (`maturin develop` for the former, `pip install -e .` for the
latter) and **each one has been observed to uninstall/clobber the other**, depending on
order and how `maturin` is invoked. Whichever ran last "won"; the other's import broke.

Root cause identified: running `maturin develop --release --manifest-path
smriti-core/Cargo.toml` **from the repo root** (`smriti/`) causes maturin to pick up the
*outer* `pyproject.toml` (CWD-based auto-discovery) instead of scoping to the Rust crate —
it built and installed a wheel literally named `smriti-0.1.0`, colliding with and
uninstalling the real editable `smriti` Python package (pip uninstalls by distribution name,
and the wheel's RECORD apparently included both `smriti_core.abi3.so` and got attributed to
the `smriti` distribution name).

**Fix applied and confirmed to correctly name the wheel (in the prior session):** run maturin
from *inside* `smriti-core/`:
```
cd smriti-core
maturin develop --release
cd ..
```
This produced a correctly-named `smriti-core-0.1.0` wheel (not colliding with `smriti`).
**However**, immediately after that + `pip install -e .`, `import smriti_core` failed again
(`ModuleNotFoundError`) — this was NOT re-isolated before the prior session ended, and per
the re-check above, both are still broken now. It's unclear whether:
- `pip install -e .`'s build-isolation step (it spins up a temp build venv per PEP 517) is
  somehow still touching the target venv's `smriti_core` install, or
- the two are landing correctly but something else (stale `__pycache__`, a second Python
  install shadowing `.venv/bin/python` via PATH, etc.) is the actual cause, or
- there's a genuine ordering dependency that hasn't been found yet.

Also seen once, likely a red herring but worth ruling out early: mixing `uv run` /
`uv pip install -e .` into the sequence appeared to re-sync the environment and strip
`smriti_core` because it's not a declared dependency in `pyproject.toml`. **Avoid `uv` for
this — use the venv's own `pip`/`python`/`maturin` directly** until the plain-pip path is
confirmed working, at minimum. (Note: a `uv.lock` file now exists in the repo, untracked —
if something has since switched this project to a `uv`-managed workflow, that itself may be
worth reconciling with this note rather than ignoring.)

## What to actually do
1. `cd ~/Code/self/smriti && source .venv/bin/activate`
2. Get a clean read on current state first:
   ```
   find .venv -iname "*smriti_core*"
   find .venv -iname "*smriti-0.1.0*" -o -iname "*smriti_core-0.1.0*"
   python -c "import smriti_core" ; echo $?
   python -c "import smriti" ; echo $?
   ```
3. Isolate the two installs completely — do them one at a time, checking imports after each,
   rather than chaining commands and only checking at the end (that's what caused most of the
   back-and-forth confusion in the prior sessions — errors were being diagnosed from combined
   output of 3-4 chained commands at once).
4. Once `cd smriti-core && maturin develop --release && cd ..` is confirmed (in isolation) to
   leave `import smriti_core` working, figure out why `pip install -e .` running afterward
   breaks it — check `pip install -e . -v` for any uninstall/RECORD messages touching
   `smriti_core`, and check whether `pip install -e .`'s build isolation
   (`--no-build-isolation` flag as a test) changes the outcome.
5. Once both imports work simultaneously, run:
   ```
   python scripts/eval_naksha_seed.py --fallback-embeddings
   ```
   and separately with real embeddings (drop the flag) once that's clean.
6. Report back: the six-section output the script prints (topology stats, self-recall,
   domain purity, edge-discovery distance check, resolution-dial sweep) — this feeds back
   into `claude/adr-smriti-integration.md` in the Naksha Claude project (Ankur has that open
   elsewhere) as the "actionable now" eval spike result. That ADR was also updated
   (2026-09-14) with a "Confidence levels" section, theoretical notes on why the HAM-R
   mechanism is non-parametric and how it relates to ANN methods (HNSW/IVF), and a cost-model
   note specifically flagging that `query_hopfield()` latency hasn't been benchmarked
   separately from the plain top-k query path — worth measuring both once this runs, plus the
   real (non-fallback) embedding pairwise-similarity distribution and resolution-dial sweep
   behavior on the actual Naksha corpus, per that ADR's open questions section.
7. Consider committing `scripts/eval_naksha_seed.py` and this file (or its resolution) once
   things work, so a future session doesn't lose this context again the way this file itself
   was lost between sessions.

## Context, if useful
- `../naksha/` is a sibling repo with the real seed data this script reads.
- `../naksha/docs/execute/Alpha_Sprint_Plan.md` and the Naksha Claude-project ADRs
  (`adr-memory-pipeline-patterns.md`, `adr-sambandh-crossover.md`,
  `adr-smriti-integration.md`) have the fuller context on why this integration test matters
  and what it's blocked/not-blocked on. Not required reading to fix the build, just useful if
  the "why" comes up.
