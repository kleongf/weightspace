# NNNN: Title

Status: draft

## Hypothesis

One falsifiable sentence.

## Kind

Fault repair, or Capability gain.

## Target

Model, revision, precision.

## Task and metric

What the Target is scored on, and how one example gets a Strict score and a Lenient score.

## Practice set

Source, size, and how it was drawn.

## Held-out set

Source, size, and how it was kept apart from the Practice set.

## Tools

What the Editor may use in a Run.

## Eval budget

Number of Practice set evaluations per Run. Default 50.

## Turn limit

Maximum Editor turns per Run. Default 60.

## Spend cap

Maximum cost per Run. Default $20. A Run stops at whichever limit it hits first, and its best Patch so far is still scored.

## Controls

Number of Random-edit controls (default 5), plus any others, such as a Fine-tune baseline.

## Success criterion

The Held-out set result, against the controls, that would confirm the hypothesis. Default: the final Patch beats the best Random-edit control, with the lower bound of a paired bootstrap 95% interval above zero, on the Strict score for Fault repair or the Lenient score for Capability gain.

## Amendments

Append-only after Run 1. Date each entry.
