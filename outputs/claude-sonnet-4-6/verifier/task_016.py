def build_pipeline(steps, config=None, intermediate_results=None):
    """
    Build and execute a processing pipeline.
    
    Args:
        steps: List of callable processing steps to apply sequentially
        config: Optional dictionary of configuration options
        intermediate_results: Optional list to collect intermediate results
        
    Returns:
        A function that accepts an input value and returns a tuple of
        (final_output, collected_results)
    """
    if config is None:
        config = {}
    
    if intermediate_results is None:
        intermediate_results = []
    
    def pipeline(input_value):
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
        
        # Return collected results excluding the initial input for cleaner output
        collected = intermediate_results[1:]  # Exclude initial input
        
        return final_output, collected
    
    return pipeline


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
    
    # Test 1: Basic pipeline
    print("Test 1: Basic pipeline")
    steps = [normalize, scale, round_values]
    config = {'scale_factor': 3.0, 'decimals': 3}
    results_collector = []
    
    pipeline = build_pipeline(steps, config, results_collector)
    
    input_data = [10, 20, 30, 40, 50]
    final_output, collected = pipeline(input_data)
    
    print(f"Input: {input_data}")
    print(f"Final output: {final_output}")
    print(f"Intermediate results: {collected}")
    print()
    
    # Test 2: Pipeline without config
    print("Test 2: Pipeline without config")
    simple_steps = [
        lambda x: [v * 2 for v in x],
        lambda x: [v + 1 for v in x],
        lambda x: sorted(x, reverse=True)
    ]
    
    pipeline2 = build_pipeline(simple_steps)
    final2, collected2 = pipeline2([1, 2, 3, 4, 5])
    
    print(f"Input: [1, 2, 3, 4, 5]")
    print(f"Final output: {final2}")
    print(f"Intermediate results: {collected2}")
    print()
    
    # Test 3: Single step pipeline
    print("Test 3: Single step pipeline")
    pipeline3 = build_pipeline([lambda x: x ** 2])
    final3, collected3 = pipeline3(5)
    
    print(f"Input: 5")
    print(f"Final output: {final3}")
    print(f"Intermediate results: {collected3}")
    print()
    
    # Test 4: Empty steps pipeline
    print("Test 4: Empty steps pipeline")
    pipeline4 = build_pipeline([])
    final4, collected4 = pipeline4("hello")
    
    print(f"Input: 'hello'")
    print(f"Final output: {final4}")
    print(f"Intermediate results: {collected4}")