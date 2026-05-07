def build_pipeline(steps, config=None, intermediate_results=None):
    if config is None:
        config = {}
    if intermediate_results is None:
        intermediate_results = []

    def pipeline(input_value):
        current_value = input_value
        for step in steps:
            # Apply the step with the current value and configuration
            current_value = step(current_value, **config)
            # Store the intermediate result
            intermediate_results.append(current_value)
        return current_value, intermediate_results

    return pipeline