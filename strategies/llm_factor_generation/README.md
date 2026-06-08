# LLM Factor Generation

This lane uses the current alpha stack and external research themes to generate
factor hypotheses that can be falsified. It does not use an LLM as a trading
oracle, and it does not promote generated text into trades.

```bash
uv run python -m strategies.llm_factor_generation.current_factor_hypothesis_templates
uv run python -m strategies.llm_factor_generation.current_factor_template_validation_queue
```

The output is a broad queue of formula-like factor ideas. A template must still
be turned into data, labels, costs, and a leakage-safe validation path before it
can become a trading strategy candidate.
