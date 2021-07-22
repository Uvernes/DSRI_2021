import math
import random


def window_slice_single_time_series(t_series, window_slice_prop=0.9):

    num_timestamps = len(t_series)
    left_out_prop = 1 - window_slice_prop
    left_out_amount = round(num_timestamps * left_out_prop)
    left_out_beginning = random.randint(0, left_out_amount)
    left_out_end = left_out_amount - left_out_beginning

    return t_series[left_out_beginning:num_timestamps - left_out_end]


def window_slicing_specified_amount(dataset, synthetic_amount, window_slice_prop=0.9, shuffle=True):
    synthetic_samples = []
    if (synthetic_amount <= 0) or (window_slice_prop <= 0) or (window_slice_prop > 1):
        return synthetic_samples
    if shuffle:
        random.shuffle(dataset)
    while True:
        for t_series in dataset:
            synthetic_samples.append(window_slice_single_time_series(t_series, window_slice_prop))
            if len(synthetic_samples) == synthetic_amount:
                return synthetic_samples


# ---------- Testing --------- #
# print("Window slicing a single time series (prop. = 0.9)....")
# time_series = [1, 2, 5, 8, 1, 3, 4, 5, 1, 4]
# print("Orginal time series:", time_series)
# print("Synthetic time series:", window_slice_single_time_series(time_series, window_slice_prop=0.9))
#
# print("\nWindow slicing a dataset (prop. = 0.9)....")
# dataset = [[1, 2, 5, 8, 1, 3, 4, 5, 1, 4], [6, 2, 5, 8, 1, 0, 9, 3, 4, 5, 1, 4, 23, -5]]
# print("Original dataset:", dataset)
# print("Synthetic dataset:")
# synthetic_dataset = window_slicing_specified_amount(dataset, 5, window_slice_prop=0.5)
# for synthetic in synthetic_dataset:
#     print(synthetic)

