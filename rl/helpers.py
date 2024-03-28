def replace_key_with_multiple(dictionary, old_key, new_key_values):
    new_dict = {}
    inserted = False
    for key, value in dictionary.items():
        if key == old_key:
            for new_key, new_value in new_key_values.items():
                new_dict[new_key] = new_value
            inserted = True
        else:
            new_dict[key] = value
    if not inserted:
        raise ValueError("The old key was not found in the dictionary.")
    return new_dict

def rolling_averages(data, interval):
    n = len(data)
    averages = []
    for i in range(0, n, interval):
        chunk = data[i:min(i+interval, n)]  # Handle the last chunk if its length is less than the interval
        if chunk:  # Check if the chunk is not empty
            averages.append(sum(chunk) / len(chunk))
    return averages