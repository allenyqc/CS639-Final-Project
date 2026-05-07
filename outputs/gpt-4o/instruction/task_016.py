def build_pipeline(steps, config=None, intermediate_results=None):
    if config is None:
        config = {}
    if intermediate_results is None:
        intermediate_results = []

    def apply_steps(input_value):
        current_value = input_value
        for step in steps:
            try:
                current_value = step(current_value, **config)
                intermediate_results.append(current_value)
            except (TypeError, ValueError) as e:
                print(f"Error applying step {step.__name__}: {e}")
                break
        return current_value

    return apply_steps, intermediate_results

# Example usage:
# def step1(x, factor=1):
#     return x * factor
# 
# def step2(x, increment=0):
#     return x + increment
# 
# pipeline, results = build_pipeline([step1, step2], config={'factor': 2, 'increment': 3})
# final_output = pipeline(5)
# print(final_output)  # Output should be 13
# print(results)      # Output should be [10, 13]