def collect_results(result, results=None):
    if results is None:
        results = []
    try:
        results.append(result)
    except AttributeError as e:
        raise TypeError("Provided results argument must be a list.") from e
    return results