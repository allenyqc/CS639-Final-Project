from typing import Callable, Iterable, Optional, Any


def build_pipeline(
    steps: Iterable[Callable],
    config: Optional[dict] = None,
    intermediates: Optional[list] = None,
) -> Callable:
    """
    Build a pipeline that applies a sequence of callables to an input value.

    Parameters
    ----------
    steps : iterable of callables
        Processing steps to apply sequentially.
    config : dict, optional
        Configuration options. Recognized keys:
            - "verbose" (bool): print progress information.
            - "stop_on_error" (bool): re-raise step exceptions (default True).
            - "step_kwargs" (dict): mapping of step name -> kwargs dict
              passed to that step.
    intermediates : list, optional
        External list to collect intermediate results. If None, a new
        list is created.

    Returns
    -------
    callable
        A pipeline function. Calling it with an input value returns a
        tuple ``(final_output, intermediates)``.
    """
    steps = list(steps)
    config = dict(config) if config else {}
    verbose = config.get("verbose", False)
    stop_on_error = config.get("stop_on_error", True)
    step_kwargs = config.get("step_kwargs", {}) or {}

    def run(value: Any):
        results = intermediates if intermediates is not None else []

        current = value
        for i, step in enumerate(steps):
            name = getattr(step, "__name__", f"step_{i}")
            kwargs = step_kwargs.get(name, {})
            try:
                if verbose:
                    print(f"[pipeline] applying {name} (step {i})")
                current = step(current, **kwargs) if kwargs else step(current)
            except Exception as exc:
                if verbose:
                    print(f"[pipeline] {name} failed: {exc}")
                if stop_on_error:
                    raise
                current = None
            results.append(current)

        return current, results

    return run