# Experiments

This directory holds human-readable investment hypotheses and any later
artifacts needed to evaluate them.

Use one directory per hypothesis once it has more than a single note:

```text
hypotheses/<hypothesis>/README.md
hypotheses/<hypothesis>/evaluate.py
```

Keep hypothesis-specific scripts and result notes in that hypothesis directory.
Keep reusable evidence data under `datasets/` and reference it from the
hypothesis.
