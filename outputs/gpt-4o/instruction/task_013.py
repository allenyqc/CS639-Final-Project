def collect_results(result, results_list=None):
    if results_list is None:
        results_list = []
    try:
        results_list.append(result)
    except AttributeError as e:
        raise TypeError("Provided results_list is not a list") from e
    return results_list