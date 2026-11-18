# 0001: Thinking-mode overrun on JSON extraction

Status: frozen

## Hypothesis

Without gradients, a Claude Opus 5.5 Editor with a Python sandbox and 30 Practice evaluations can write a Patch that raises Qwen3.5-0.8B's Strict score on JSON extraction above the best of 5 Random-edit controls on 100 Held-out examples, scored with thinking on, sampled decoding and a 2,048-token limit.

The Fault: in the sampled survey (`surveys/2026-11-17-Qwen3.5-0.8B-json-extraction-on-sampled.md`) the Baseline scored Strict 0.45 and Lenient 1.00. 20 of 40 answers ran out of tokens while still reasoning; 6 of those were exact loops and 14 kept deliberating, often over whether a code fence counts as "other text". 18 of the 20 finished answers were exactly right.

## Kind

Fault repair. The Lenient score is already 1.00, so the Target knows every answer and fails only by not finishing, or by wrapping the JSON in a code fence.

## Target

Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`, bf16, loaded from `models/Qwen3.5-0.8B` with mlx 0.32.4 and mlx-lm 0.32.2.

## Task and metric

The `json-extraction` task in `src/weightspace/tasks.py`, with thinking on. Decoding is sampled (temperature 0.6, top-p 0.95, top-k 20, seed 0) with a 2,048-token limit and batch size 4.

- Strict score: 1 when the answer finished and the text after `</think>` parses as JSON exactly equal to the reference; 0 otherwise. A code fence scores 0.
- Lenient score: the fraction of the 4 field values that appear anywhere in the output.

Secondary result, not part of the Success criterion: GSM8K carry-over. The final Patch and the Baseline are each scored on 50 GSM8K Held-out examples with the same decoding, to show whether the Patch shortens reasoning in general or only for JSON. Also reported: cut-off rate, loop-flag rate, mean tokens and Patch size.

## Practice set

The first 160 examples of `json_examples("practice", ...)`, generated from the seed `"practice"` with the same 4 sentence templates and name lists as every split. The Editor sees all 160 prompts and references in the sandbox. `evaluate` scores up to 16 example ids the Editor chooses, and the first 16 when it chooses none.

## Held-out set

The first 100 examples of `json_examples("held-out", ...)`, generated from the seed `"held-out"`. None of the 160 Practice prompts or 40 survey prompts appear among them, checked on 2026-11-17 and enforced by `tests/test_tasks.py`. The harness generates them only at scoring time and never writes them into the run folder (ADR 0002). They share the Practice set's templates, so they test new records, not new phrasings.

For the secondary result: the first 50 examples of the GSM8K test split, shuffled with seed 0.

## Tools

Exactly two tools, served by the local MCP server of ADR 0002. There is no fine-tune tool and no web access.

1. A persistent Python sandbox. The Target's weights are live mlx arrays there, alongside numpy, mlx, the tokenizer, and the Practice set's prompts and references. The Editor may run forward passes and generate text. Gradients are banned: any use of `mx.grad`, `mx.value_and_grad` or an `mlx.optimizers` optimizer voids the Run, checked by scanning the transcript. Each sandbox call times out after 10 minutes.
2. `evaluate`, which scores the current weights on 16 Practice examples and returns each example's Strict score, Lenient score, whether it was cut off, and its text. Each call spends one evaluation.

The sandbox and `evaluate` share one copy of the model, so `evaluate` always scores exactly what the Editor edited. Because the sandbox can generate, the Eval budget limits scored feedback, not compute; the Turn limit and the sandbox timeout limit compute.

## Eval budget

30 calls to `evaluate`, each scoring 16 Practice examples (about 3.5 hours).

## Turn limit

60 Editor turns, enforced by the MCP server.

## Spend cap

$20 per Run, measured by the cost the Claude CLI reports and passed as `--max-budget-usd`. If the dry run shows that flag does not stop a subscription session, the MCP server refuses tool calls once reported cost passes $20. A Run stops at whichever limit it hits first, and its best Patch so far is still scored.

## Controls

5 Random-edit controls, seeds 0 to 4. No Fine-tune baseline in this Experiment.

## Success criterion

Run 1 decides the verdict. Later Runs are exploratory and reported, but they do not change it.

The final Patch is the evaluated state with the highest Practice Strict score, the later one on a tie. Edits made after the last `evaluate` call never count.

The hypothesis is confirmed when the final Patch's Held-out Strict score beats the best of the 5 Random-edit controls, with the lower bound of a paired bootstrap 95% interval above zero. Any gradient use voids the Run. The final Patch against the Baseline is also reported.

## Amendments

Append-only after Run 1. Date each entry.

2026-11-18, before Run 1. The dry run settled the Spend cap question: `--max-budget-usd` stops a subscription session, which ends with `error_max_budget_usd`. Claude checks the cap after each model turn, so a Run can pass the cap by one turn's cost (a $0.01 cap stopped at $0.07). The MCP server's fallback check is not needed.
