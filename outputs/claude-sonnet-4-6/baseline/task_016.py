def build_pipeline(steps, config=None, intermediate_results=None):
    """
    Build and execute a processing pipeline.
    
    Args:
        steps: List of callable processing steps to apply sequentially
        config: Optional dictionary of configuration options
        intermediate_results: Optional list to collect intermediate results
        
    Returns:
        A function that accepts an input value and returns (final_output, collected_results)
    """
    if config is None:
        config = {}
    
    if intermediate_results is None:
        intermediate_results = []
    
    def execute(input_value):
        """
        Execute the pipeline on the given input value.
        
        Args:
            input_value: The initial value to process through the pipeline
            
        Returns:
            Tuple of (final_output, collected_results)
        """
        # Clear previous results if reusing the list
        intermediate_results.clear()
        
        current_value = input_value
        
        # Store the initial input
        intermediate_results.append(current_value)
        
        for step in steps:
            try:
                # Check if the step accepts config as a keyword argument
                if config:
                    import inspect
                    sig = inspect.signature(step)
                    params = sig.parameters
                    
                    # If step accepts **kwargs or 'config' parameter, pass config
                    accepts_config = (
                        'config' in params or
                        any(
                            p.kind == inspect.Parameter.VAR_KEYWORD
                            for p in params.values()
                        )
                    )
                    
                    if accepts_config:
                        current_value = step(current_value, config=config)
                    else:
                        current_value = step(current_value)
                else:
                    current_value = step(current_value)
                    
            except TypeError:
                # Fallback: call step without config if there's a TypeError
                current_value = step(current_value)
            
            # Store intermediate result after each step
            intermediate_results.append(current_value)
        
        # The final output is the last processed value
        final_output = current_value
        
        # Return collected results excluding the initial input (only step outputs)
        collected_results = intermediate_results[1:]
        
        return final_output, collected_results
    
    return execute


# Example usage and testing
if __name__ == "__main__":
    # Define some sample processing steps
    def normalize(x, config=None):
        """Normalize values to [0, 1] range."""
        if isinstance(x, list):
            min_val = min(x)
            max_val = max(x)
            if max_val == min_val:
                return [0.0] * len(x)
            return [(v - min_val) / (max_val - min_val) for v in x]
        return x
    
    def scale(x, config=None):
        """Scale values by a factor from config."""
        factor = config.get('scale_factor', 2.0) if config else 2.0
        if isinstance(x, list):
            return [v * factor for v in x]
        return x * factor
    
    def round_values(x, config=None):
        """Round values to specified decimal places."""
        decimals = config.get('decimals', 2) if config else 2
        if isinstance(x, list):
            return [round(v, decimals) for v in x]
        return round(x, decimals)
    
    # Test 1: Basic pipeline without config
    print("Test 1: Basic pipeline")
    steps = [normalize, scale, round_values]
    pipeline = build_pipeline(steps)
    
    input_data = [10, 20, 30, 40, 50]
    final, results = pipeline(input_data)
    
    print(f"Input: {input_data}")
    print(f"After normalize: {results[0]}")
    print(f"After scale: {results[1]}")
    print(f"Final output: {final}")
    print()
    
    # Test 2: Pipeline with config
    print("Test 2: Pipeline with config")
    config = {'scale_factor': 3.0, 'decimals': 3}
    intermediate = []
    pipeline_with_config = build_pipeline(steps, config=config, intermediate_results=intermediate)
    
    final2, results2 = pipeline_with_config(input_data)
    print(f"Input: {input_data}")
    print(f"Intermediate results: {results2}")
    print(f"Final output: {final2}")
    print()
    
    # Test 3: Simple numeric transformations
    print("Test 3: Simple numeric transformations")
    simple_steps = [
        lambda x: x + 10,
        lambda x: x * 2,
        lambda x: x - 5
    ]
    simple_pipeline = build_pipeline(simple_steps)
    final3, results3 = simple_pipeline(5)
    
    print(f"Input: 5")
    print(f"Step results: {results3}")
    print(f"Final output: {final3}")
    print()
    
    # Test 4: String processing pipeline
    print("Test 4: String processing pipeline")
    string_steps = [
        str.strip,
        str.lower,
        lambda s: s.replace(' ', '_')
    ]
    string_pipeline = build_pipeline(string_steps)
    final4, results4 = string_pipeline("  Hello World  ")
    
    print(f"Input: '  Hello World  '")
    print(f"Step results: {results4}")
    print(f"Final output: '{final4}'")