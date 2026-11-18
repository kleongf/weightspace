---
name: new-experiment
description: Start a weightspace Experiment by interviewing the user until its card is filled and frozen. Use when the user wants to start, design, or plan an Experiment, or fill in a card.
---

# New experiment

An Experiment's card is its contract. Every section holds a decision the user confirmed before Run 1, so a result can't quietly change what it set out to test. Use the vocabulary in `GLOSSARY.md`. When the user says a word the glossary lists under _Avoid_, name the canonical term and confirm which one they mean.

1. **Create the card.** Choose a short slug from the user's idea and run `uv run ws new <slug>`. If a draft for this idea already exists in `uv run ws list`, continue that card.
   Done when one draft card exists for the idea.

2. **Grill the card.** Treat the template's sections as a design tree. Hypothesis and Kind come first, because Target, task and Success criterion depend on them. Each round, ask every section whose prerequisites are settled, numbered, each with your recommended answer, and worded so that "yes" accepts the recommendation. Look up facts yourself: Baseline scores, dataset sizes and what the code does. The user decides. Where the template states a default, recommend that default unless the hypothesis needs something else.
   Done when every section except Amendments holds an answer the user confirmed in this conversation.

3. **Write the answers in.** Replace each section's template text with the confirmed decision, in concrete numbers and names. A new term the user settles goes into `GLOSSARY.md` straight away.
   Done when `uv run ws list` shows `unfilled=0` for this card.

4. **Freeze.** Show the user the finished card, then run `uv run ws freeze <number>` and `uv run pytest -q`. Commit the card only when the user says so. Later changes go in Amendments, dated.
   Done when the card reads `Status: frozen` and the tests pass.
