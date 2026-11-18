# Experiment ideas

Candidates for later Experiments. Start one with `uv run ws new <slug>` and delete its line here.

- **Gradients allowed.** Repeat a finished Experiment with gradients permitted in the sandbox, to measure how much they add over gradient-free editing.
- **Fine-tune baseline.** A LoRA trained on the same Practice set and budget, as a control. It needs training targets, such as the Target's own correct answers collected by sampling.
- **Short-answer knowledge.** A Capability gain on short-answer, chosen after 0001.
- **How many Random-edit controls.** Score one finished Run's final Patch against 1, 3, 5 and 10 controls, to see how much the count changes the verdict and what it costs in scoring time.
