def build_pipeline(steps, config=None, intermediate_results=None):
    if config is None:
        config = {}
    if intermediate_results is None:
        intermediate_results = []

    def pipeline(input_value):
        current_value = input_value
        for step in steps:
            try:
                current_value = step(current_value, **config)
                intermediate_results.append(current_value)
            except TypeError as e:
                print(f"TypeError encountered in step {step.__name__}: {e}")
                raise
            except ValueError as e:
                print(f"ValueError encountered in step {step.__name__}: {e}")
                raise
        return current_value, intermediate_results

    return pipeline