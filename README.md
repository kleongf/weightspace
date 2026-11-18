# weightspace

A lab for testing one question: can an LLM **Editor** improve a small **Target** model on a chosen task by writing its weights directly, with no training loop?

Each claim this repo makes comes from an **Experiment** whose card was written before its first **Run**. See [GLOSSARY.md](GLOSSARY.md) for the vocabulary.

## Layout

```
experiments/         one folder per Experiment
  _template/card.md  copy this to start a new Experiment
  NNNN-slug/
    card.md          hypothesis, setup and success criterion, frozen before Run 1
    runs/            one record per Run (raw transcripts and weights are git-ignored)
surveys/             Baseline scores on candidate tasks, used to choose an Experiment's task
src/weightspace/     the harness: tasks and scorers, Target loading, Patches and controls, and the Editor's MCP server
tests/               checks that keep Experiments fair (card rules, Eval budget, Held-out set isolation)
```

## Rules of a fair Experiment

1. Write and commit the card before Run 1. After that, change it only by appending to its Amendments section.
2. The Editor never sees the Held-out set. It runs outside this repo with only its tools (see [ADR 0002](docs/adr/0002-isolated-editor.md)).
3. Every Run has a hard Eval budget and Turn limit. Claude Runs also carry a Spend cap.
4. A result counts only if it meets the card's Success criterion against its Random-edit controls on the Held-out set.

`uv run pytest` checks the rules that can be checked automatically, including the sandbox's isolation (it loads the Target, about 30 seconds). `uv run pytest -m slow` adds the end-to-end Patch checks (about 6 minutes). `uv run pytest -m editor` starts each Editor CLI for real on your subscriptions and checks it reaches both tools and nothing else (about 2 minutes).

## Commands

```
uv run ws new <slug>         start an Experiment from the card template
uv run ws freeze <number>    lock a filled card before Run 1 (refuses unfilled sections)
uv run ws list               every Experiment with its status, Run count and unfilled sections
uv run ws survey             score the Baseline on every candidate task, with thinking off and on
uv run ws run <number>       start a Run: a headless Editor with only the sandbox and evaluate
                             (--editor claude|codex, default claude)
```

`ws survey` writes a table to `surveys/` and every completion to `surveys/raw/` (git-ignored). It expects the Target in `models/Qwen3.5-0.8B`, downloaded from Hugging Face.

`ws run` reads the Experiment's `run.json` and `editor.md`, works in `~/weightspace-runs/` (outside the repo, per ADR 0002), and copies the record into the Experiment's `runs/` when the Editor stops. `ws run <number> --dry-run` checks the plumbing with 2 turns and one 2-example evaluation, and keeps nothing in the repo.

A real Run needs the Mac on power and 3 GB of free disk. Only the best Patch so far is kept, and `evaluate` refuses a Patch that would leave less than 1 GB free.

Two Editors are supported, each on its own subscription with no API key. `claude` runs `claude -p`. `codex` runs `codex exec` under `sandbox-exec`, with a fresh `CODEX_HOME` per Run that borrows only your login. `ws` finds Codex at `$WS_CODEX`, then on `PATH`, then inside the ChatGPT app. Both reach the same local MCP server over HTTP on 127.0.0.1, at a secret path that changes every Run.

## Machine

This repo targets an Apple M2 with 16 GB of RAM and no CUDA. Evaluation runs on MLX.
