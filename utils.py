import numpy as np

def find_lower_nearest_number(np_arr, target_val):
    sorted_indices = np.argsort(np_arr)
    idx = sorted_indices[0]
    for i in sorted_indices:
        if np_arr[i] <= target_val:
            idx = i
    return idx
