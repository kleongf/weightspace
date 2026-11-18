# weightspace

The vocabulary for experiments where an LLM changes another model's behaviour by reading and writing its weights directly.

## The method

**Weight-space programming**:
Changing a model's behaviour by having an Editor read its weights and write a Patch from a plain-language goal, with no gradient-based training.
_Avoid_: fine-tuning, model editing, weight editing

**Target**:
The model whose weights the Editor reads and patches.
_Avoid_: subject model, student

**Baseline**:
The Target's unmodified weights, and the scores they get.
_Avoid_: original model, base model

**Editor**:
The LLM agent that inspects the Target and produces Patches.
_Avoid_: agent, programmer, teacher

**Patch**:
A set of changes to the Target's weights, expressed relative to the Baseline.
_Avoid_: edit, diff, delta

## What a Patch is for

**Fault**:
A measurable failure that stops the Target from showing an ability it already has, such as looping or a broken output format. A Patch that raises only the Strict score repaired a Fault.
_Avoid_: bug, glitch

**Capability**:
An ability on a task that the Baseline lacks even when its Faults are set aside. A Patch that raises the Lenient score added Capability.
_Avoid_: skill, knowledge

## Evaluation

**Practice set**:
The examples the Editor may score its Patches against during a Run.
_Avoid_: training set, dev set

**Held-out set**:
The examples used only to score a Run's final Patch, never visible to the Editor.
_Avoid_: test set, validation set

**Eval budget**:
The number of Practice set evaluations the Editor may spend in one Run.
_Avoid_: pass budget, forward-pass budget

**Strict score**:
A score that counts an answer only when the output is well-formed and finishes.
_Avoid_: exact score, raw score

**Lenient score**:
A score that judges only the answer, ignoring loops, truncation and formatting.
_Avoid_: soft score, relaxed score

**Random-edit control**:
A Patch that touches the same number of weights as the Editor's final Patch, chosen at random, and scored on the Held-out set for comparison.
_Avoid_: null patch, random baseline

**Carry-over**:
A Patch's score on a second task it was not written for, next to the Baseline's, showing whether its effect spreads beyond the Experiment's task.
_Avoid_: transfer, side effect

**Fine-tune baseline**:
A LoRA fine-tune of the Target with a budget matched to a Run, scored alongside the Runs for comparison.
_Avoid_: training baseline, gradient baseline

## Organising the work

**Experiment**:
A hypothesis plus the fixed setup used to test it, recorded in a card before the first Run.
_Avoid_: study, trial

**Run**:
One Editor session within an Experiment, from Baseline to final Patch.
_Avoid_: attempt, episode, session

**Survey**:
A scoring of the Baseline on candidate tasks, used to choose an Experiment's task before its card is frozen.
_Avoid_: benchmark, sweep
