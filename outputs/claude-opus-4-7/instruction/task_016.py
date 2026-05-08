from typing import Any, Callable, Iterable, Optional


class PipelineStepError(RuntimeError):
    """Raised when a pipeline step fails to execute."""

    def __init__(self, index: int, step_name: str, original: BaseException) -> None:
        super().__init__(
            f"Step {index} ({step_name!r}) failed: "
            f"{type(original).__name__}: {original}"
        )
        self.index = index
        self.step_name = step_name
        self.original = original


def build_pipeline(
    steps: Iterable[Callable[[Any], Any]],
    initial_value: Any = None,
    config: Optional[dict] = None,
    intermediates: Optional[list] = None,
) -> tuple:
    """
    Apply a sequence of callables to ``initial_value`` and collect intermediates.

    Parameters
    ----------
    steps : iterable of callables
        Each callable takes the current value (and optionally a ``config`` kwarg)
        and returns the next value.
    initial_value : Any
        The starting value fed into the first step.
    config : dict, optional
        Configuration options. If a step accepts a ``config`` keyword argument
        it will be passed in. Defaults to an empty dict.
    intermediates : list, optional
        If provided, intermediate results are appended to it. Otherwise a new
        list is created.

    Returns
    -------
    (final_output, intermediates) : tuple
    """
    if config is None:
        config = {}
    if intermediates is None:
        intermediates = []

    verbose = bool(config.get("verbose", False))
    log_path = config.get("log_path")
    stop_on_error = bool(config.get("stop_on_error", True))

    steps_seq = list(steps)
    current = initial_value

    log_ctx = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        with _maybe_context(log_ctx) as log_file:
            for index, step in enumerate(steps_seq):
                step_name = getattr(step, "__name__", repr(step))
                try:
                    if _accepts_kwarg(step, "config"):
                        current = step(current, config=config)
                    else:
                        current = step(current)
                except (TypeError, ValueError, KeyError, AttributeError,
                        ArithmeticError, LookupError) as exc:
                    err = PipelineStepError(index, step_name, exc)
                    if log_file is not None:
                        log_file.write(f"{err}\n")
                    if stop_on_error:
                        raise err from exc
                    intermediates.append(None)
                    continue

                intermediates.append(current)
                if verbose:
                    msg = f"[pipeline] step {index} ({step_name}) -> {current!r}"
                    if log_file is not None:
                        log_file.write(msg + "\n")
                    else:
                        print(msg)
    finally:
        # Context manager handles closing; nothing extra needed.
        pass

    return current, intermediates


def _accepts_kwarg(func: Callable, name: str) -> bool:
    """Return True if ``func`` accepts a keyword argument ``name``."""
    import inspect
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if name in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _maybe_context(resource):
    """Wrap a resource in a context manager, or provide a null one."""
    if resource is None:
        return _NullContext()
    return resource