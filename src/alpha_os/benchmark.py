from __future__ import annotations


class Benchmark:
    """Reserved for trading comparison references, not evaluation settings."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Benchmark is reserved for trading comparison references. "
            "Use EvaluationSpec for evaluation settings."
        )
