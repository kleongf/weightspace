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
src/weightspace/     the harness: tasks and scorers, Target loading, and later the sandbox, evaluator and controls
tests/               checks that keep Experiments fair (card rules, Eval budget, Held-out set isolation)
```

## Rules of a fair Experiment

1. Write and commit the card before Run 1. After that, change it only by appending to its Amendments section.
2. The Editor never sees the Held-out set. It runs outside this repo with only its tools (see [ADR 0002](docs/adr/0002-isolated-editor.md)).
3. Every Run has a hard Eval budget, turn limit and spend cap.
4. A result counts only if it meets the card's Success criterion against its Random-edit controls on the Held-out set.

`uv run pytest` checks the rules that can be checked automatically.

## Commands

```
uv run ws new <slug>         start an Experiment from the card template
uv run ws freeze <number>    lock a filled card before Run 1 (refuses unfilled sections)
uv run ws list               every Experiment with its status, Run count and unfilled sections
uv run ws survey             score the Baseline on every candidate task, with thinking off and on
```

`ws survey` writes a table to `surveys/` and every completion to `surveys/raw/` (git-ignored). It expects the Target in `models/Qwen3.5-0.8B`, downloaded from Hugging Face.

## Machine

This repo targets an Apple M2 with 16 GB of RAM and no CUDA. Evaluation runs on MLX.
