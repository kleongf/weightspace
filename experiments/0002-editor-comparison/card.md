# 0002: Editor comparison

Status: frozen

## Hypothesis

With the same prompt, tools and limits, Claude Opus 5.5 and GPT-6.2 Astra Editors, both at effort high, reach different Held-out Strict scores with their final Patches on card 0001's thinking-mode overrun Fault, over 3 Runs each. Neither Editor is predicted to win. With 3 Runs per Editor, only a large difference can show.

Runs go in this order, all from one commit: claude-01, codex-01, claude-02, codex-02, claude-03, codex-03. Card 0001's Run 1 is separate and does not count here.

A Run stopped by anything other than its own limits or its Editor ending it (a subscription usage limit, a crash, a network drop, the Mac sleeping) is rerun, however much budget it had spent. The stopped record is kept and marked void, each slot gets at most 2 reruns, and each rerun gets a dated Amendment.

## Kind

Fault repair, as in card 0001. The Editors are what is compared, but the task is still repairing the overrun Fault, so the Strict score is primary.

## Target

Same as card 0001.

Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`, bf16, loaded from `models/Qwen3.5-0.8B` with mlx 0.32.4 and mlx-lm 0.32.2.

## Task and metric

Same as card 0001: the `json-extraction` task with thinking on, sampled decoding (temperature 0.6, top-p 0.95, top-k 20, seed 0), a 2,048-token limit and batch size 4.

- Strict score: 1 when the answer finished and the text after `</think>` parses as JSON exactly equal to the reference; 0 otherwise. A code fence scores 0.
- Lenient score: the fraction of the 4 field values that appear anywhere in the output.

Secondary results, not part of the Success criterion:

- GSM8K carry-over: the Baseline and all 6 final Patches, each scored on 50 GSM8K Held-out examples with the same decoding.
- Per Run: input, cached and output tokens, wall time, evaluations used, turns used, Patch size, cut-off rate and loop-flag rate.

## Practice set

Same as card 0001.

The first 160 examples of `json_examples("practice", ...)`, generated from the seed `"practice"` with the same 4 sentence templates and name lists as every split. The Editor sees all 160 prompts and references in the sandbox. `evaluate` scores up to 16 example ids the Editor chooses, and the first 16 when it chooses none.

## Held-out set

Same as card 0001.

The first 100 examples of `json_examples("held-out", ...)`, generated from the seed `"held-out"`. None of the 160 Practice prompts or 40 survey prompts appear among them, checked on 2026-11-17 and enforced by `tests/test_tasks.py`. The harness generates them only at scoring time and never writes them into the run folder (ADR 0002). They share the Practice set's templates, so they test new records, not new phrasings.

For the secondary result: the first 50 examples of the GSM8K test split, shuffled with seed 0.

## Tools

The same two tools as card 0001, served by one local MCP server per Run that `ws` starts outside any sandbox (ADR 0002): the persistent Python sandbox and `evaluate`. Both Editors get the same prompt, `editor.md`, copied unchanged from card 0001.

- Claude: `claude -p` with every built-in tool switched off.
- Codex: `codex exec` with the shell, web search, subagents, skills, plugins and memories switched off. Three built-ins cannot be switched off: `apply_patch`, `view_image` and the JavaScript host that carries MCP calls. The whole Codex process runs under `sandbox-exec`, which lets them reach only the Run's own `editor` and `codex-home` folders.

Gradients are banned for both, as in card 0001.

Only the Patch that can still become the final Patch is kept on disk: the best Practice Strict so far, the later on a tie. Every evaluation's outputs are kept. `evaluate` refuses, spending no evaluation, a Patch whose uncompressed size (16 bytes per changed weight) would leave less than 1 GB of disk free. A final Patch over 50 MB stays in `~/weightspace-runs/`, and the repo's record names it by size and SHA-256. `ws run` will not start a real Run on battery or with less than 3 GB free.

## Eval budget

Same as card 0001.

30 calls to `evaluate`, each scoring 16 Practice examples (about 3.5 hours).

## Turn limit

Same as card 0001.

60 Editor turns, enforced by the MCP server.

## Spend cap

Claude: $20 per Run, passed as `--max-budget-usd`, as a safety stop only. Codex reports no cost and has no cap. The shared hard limits are the Eval budget and the Turn limit. Each Run's input, cached and output tokens and its wall time are reported.

## Controls

5 Random-edit controls per Run, seeds 0 to 4, each matched to that Run's final Patch in size and tensors. They feed each Run's secondary verdict, the same test as card 0001's Run 1. No Fine-tune baseline.

## Success criterion

Each Run's final Patch is chosen as in card 0001: the evaluated state with the highest Practice Strict score, the later one on a tie. Edits made after the last `evaluate` call never count. Any gradient use voids the Run, and a voided Run is rerun like an interrupted one.

The hypothesis is confirmed when all 3 of one Editor's final Patches score higher Held-out Strict than all 3 of the other's. If the Editors were equally good, this would happen by luck 1 time in 10 (2 of the 20 ways to split 6 Runs into two groups of 3), so a confirmation is evidence at that strength, not proof. Any overlap means no difference is shown. A tie between two Runs from different Editors counts as overlap.

Also reported, without changing the verdict:

- Each Editor's mean Held-out Strict and the spread of its 3 Runs.
- The difference in means, with an example-level paired bootstrap 95% interval. It describes these 6 Patches only, because it ignores Run-to-Run luck.
- How many of each Editor's Runs pass card 0001's test: Held-out Strict above the best of its 5 Random-edit controls, with the lower bound of a paired bootstrap 95% interval above zero.
- Each final Patch against the Baseline.

## Amendments

Append-only after Run 1. Date each entry.
