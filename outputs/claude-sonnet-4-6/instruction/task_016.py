import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def build_pipeline(
    steps: list[Callable],
    config: Optional[dict] = None,
    intermediate_results: Optional[list] = None,
) -> tuple[Any, list]:
    """
    Apply a sequence of processing steps to an input value, collecting intermediate results.

    Args:
        steps: A list of callables to apply sequentially. Each callable receives
               the current value and the config dict, and returns a transformed value.
        config: Optional dictionary of configuration options passed to each step.
        intermediate_results: Optional list to collect intermediate results.
                              If None, a new list is created internally.

    Returns:
        A tuple of (final_output, collected_results) where collected_results contains
        the output of every step including the initial input.

    Raises:
        TypeError: If steps is not a list or any step is not callable.
        ValueError: If steps is empty.
    """
    if not isinstance(steps, list):
        raise TypeError(f"steps must be a list, got {type(steps).__name__}")
    if not steps:
        raise ValueError("steps must contain at least one callable")
    for idx, step in enumerate(steps):
        if not callable(step):
            raise TypeError(f"Step at index {idx} is not callable: {step!r}")

    # Best practice #1: never use mutable defaults — create fresh objects when None
    if config is None:
        config = {}
    if intermediate_results is None:
        intermediate_results = []

    # Determine the starting value from config, defaulting to None
    current_value: Any = config.get("initial_value", None)

    # Record the initial value before any processing
    intermediate_results.append(current_value)

    for step_index, step in enumerate(steps):
        step_name = getattr(step, "__name__", repr(step))
        logger.debug("Running step %d (%s) with value: %r", step_index, step_name, current_value)

        try:
            current_value = step(current_value, config)
        except TypeError as exc:
            # Raised when the step signature doesn't match the arguments we pass
            logger.error(
                "Step %d (%s) raised TypeError: %s", step_index, step_name, exc
            )
            raise
        except ValueError as exc:
            # Raised when a step receives a value it cannot process
            logger.error(
                "Step %d (%s) raised ValueError: %s", step_index, step_name, exc
            )
            raise

        intermediate_results.append(current_value)
        logger.debug("Step %d (%s) produced: %r", step_index, step_name, current_value)

    final_output = current_value
    return final_output, intermediate_results


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    def double(value: Any, cfg: dict) -> Any:
        factor = cfg.get("factor", 2)
        return value * factor

    def add_offset(value: Any, cfg: dict) -> Any:
        offset = cfg.get("offset", 10)
        return value + offset

    def to_string(value: Any, cfg: dict) -> str:
        prefix = cfg.get("prefix", "result")
        return f"{prefix}={value}"

    pipeline_steps = [double, add_offset, to_string]
    pipeline_config = {"initial_value": 5, "factor": 3, "offset": 7, "prefix": "output"}

    output, results = build_pipeline(pipeline_steps, config=pipeline_config)

    print("Intermediate results:", results)
    print("Final output:", output)
    # Expected: 5 -> 15 -> 22 -> "output=22"