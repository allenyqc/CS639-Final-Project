from typing import Callable, Any


def build_pipeline(
    steps: list[Callable],
    config: dict | None = None,
    intermediates: list | None = None,
) -> Callable:
    """
    Build a processing pipeline from a list of callable steps.

    Args:
        steps: A list of callables to apply sequentially.
        config: Optional dictionary of configuration options. If a step accepts
            a `config` keyword argument, it will be passed.
        intermediates: Optional list to collect intermediate results. If None,
            a new list is created.

    Returns:
        A function that takes an input value and returns a tuple of
        (final_output, collected_intermediates).
    """
    config = config if config is not None else {}
    collected = intermediates if intermediates is not None else []

    def run(value: Any) -> tuple[Any, list]:
        current = value
        for step in steps:
            try:
                current = step(current, config=config)
            except TypeError:
                current = step(current)
            collected.append(current)
        return current, collected

    return run