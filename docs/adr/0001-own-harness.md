# Build our own harness instead of using wsplab

wsplab already wraps an Editor, a weight sandbox, an Eval budget and a Random-edit control, but it is an unofficial alpha, not affiliated with the original authors, and running it means executing third-party code on the machine that holds the Target's weights. We build our own small harness so we can audit every fairness guarantee, back each one with a test, and add tools later (a fine-tune tool, for example) without forking someone else's design. wsplab's source stays a reference for its control and statistics, never a dependency.
