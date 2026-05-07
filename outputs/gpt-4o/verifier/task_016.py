def build_pipeline(steps, config=None, collect_results=None):
    if config is None:
        config = {}
    if collect_results is None:
        collect_results = []

    def pipeline(input_value):
        current_value = input_value
        for step in steps:
            current_value = step(current_value, **config)
            collect_results.append(current_value)
        return current_value, collect_results

    return pipeline