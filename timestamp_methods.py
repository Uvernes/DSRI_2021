from statistics import mean, stdev
from numpy import random
import math

"""
Parameters
-----------

timestamps_set - 2D list, where inner lists store the timestamp values for some time series.

Returns
-------

 1) Mean difference between two adjacent timestamps, considering timestamps in all inner lists

 2) Variance between two adjacent time stamps, considering timestamps in all inner lists
 
 3) Min difference between two adjacent timestamps found
 
 4) Max difference between two adjacent timestamps found
 
"""


def summary_stats_of_timestamp_differences(timestamps_set):
    timestamp_differences = []
    min = math.inf
    max = -math.inf
    for inner_list in timestamps_set:
        for i in range(1, len(inner_list)):
            dif = inner_list[i] - inner_list[i-1]
            timestamp_differences.append(dif)
            if dif > max:
                max = dif
            if dif < min:
                min = dif
    x_bar = mean(timestamp_differences)
    return x_bar, stdev(timestamp_differences, x_bar), min, max


def summary_stats_of_starting_timestamps(timestamps_set):
    starting_timestamps = []
    min = math.inf
    max = -math.inf
    for inner_list in timestamps_set:
        start = inner_list[0]
        starting_timestamps.append(start)
        if start > max:
            max = start
        if start < min:
            min = start
    x_bar = mean(starting_timestamps)
    return x_bar, stdev(starting_timestamps, x_bar), min, max


"""

This method takes in a dataset of time series (synthetic) and returns a 2D list of timestamps. Inner list i becomes
the synthetic timestamps for time series i.

Parameters
-----------

dataset        - Dataset of time series (synthetic) to create set of timestamps for
timestamps_set - Set of all existing timestamps data. Statistics retrieved from this set are used to create
                 synthetic timestamps

Return
------
2D list of synthetic timestamps 
"""


def create_synthetic_timestamps(dataset, timestamps_set):

    mean_dif, var_dif, min_dif, max_dif = summary_stats_of_timestamp_differences(timestamps_set)
    mean_start, var_start, min_start, max_start = summary_stats_of_starting_timestamps(timestamps_set)
    synthetic_timestamps_set = []

    for t_series in dataset:
        differences = random.normal(mean_dif, var_dif, len(t_series) - 1)
        # Line below ensure starting time falls within min and max starting times observed
        starting_timestamp = min(max(random.normal(mean_start, var_start), min_start), max_start)
        temp_timestamps = [starting_timestamp]
        for i in range(len(differences)):
            # Line below ensures difference falls within min and max differences observed
            dif = min(max(differences[i], min_dif), max_dif)
            temp_timestamps.append(temp_timestamps[-1] + dif)
        synthetic_timestamps_set.append(temp_timestamps)

    return synthetic_timestamps_set


# ----------- Testing ------------- #

# # Testing summary stat methods
# timestamps_set = [list(range(1, 10)), list(range(5, 24, 2)), list(range(3, 7, 3))]
# print("Timestamps set:")
# print(timestamps_set)
# print("Stats regarding differences: ", summary_stats_of_timestamp_differences(timestamps_set))
# print("Stats regarding starting times: ", summary_stats_of_starting_timestamps(timestamps_set))
#
# # Testing synthetic timestamps method
# dataset = [[1, 1, 1, 1], [1, 2, 3, 4, 5], [2, 3, 4]]
# synthetic_timestamps_set = create_synthetic_timestamps(dataset, timestamps_set)
# print("\nDataset: ")
# print(dataset)
# print("Synthetic timestamps for dataset:")
# for timestamps in synthetic_timestamps_set:
#     print(timestamps)

