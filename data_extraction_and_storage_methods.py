import os
import random

START_PATH = r"C:\Users\uvern\Dropbox\My PC (LAPTOP-554U8A6N)\Documents\DSRI\Data\usneedle_data\usneedle_data"

"""
This method extracts the time series data from a single MHA file.

Parameters
----------
filename - mha file to extract time series from. Needs to be extended to NeedleAndProbeToReference files later

Return
------
List corresponding to time series of 12 variables (9 from R matrix, 3 from translation vector).
List is of the following form:
[R_11, R_12, R_13, t_1, ... , R_31, R_32, R_33, t_3]

"""


def get_single_time_series_from_file(filename):
    time_series = []
    f = open(filename)
    for line in f:
        if ("ReferenceTransform" not in line) and ("SequenceTransform" not in line):
            continue
        split_line = line.replace("=", " ").split()
        if "Status" in line:
            if split_line[1] != "OK":
                time_series.pop()
            continue
        temp_list = []
        for i in range(1,13):
            temp_list.append(float(split_line[i]))
        time_series.append(temp_list)
    f.close()
    return time_series


"""
This method extracts all time series data available for a particular skill level (e.g novice).
**Later extensions: 
  1) Optional parameter to extract only IP or OOP data
  2) Optional parameter to exclude final trials for novices (at this point they may no longer be novices)

Parameters
----------
skill level - Particular skill level to extract all available time series data for

filename - Filename of the type of sequence being considered (e.g NeedleTipToReference-Sequence.mha)

return
------
Time series dataset (subset of all data) for a particular skill level
"""


def get_subset_of_dataset(skill_level, filename):
    dataset = []
    if skill_level not in ["SmallNovice", "Novice", "Expert"]:
        return dataset
    directories = os.listdir(START_PATH + "\\" + skill_level + "Data")
    for dir in directories:
        if "sqbr" in dir:
            continue
        # print(dir)
        # print(START_PATH + "\\" + skill_level + "Data" + "\\" + dir + "\\" + sequence_type)
        dataset.append(get_single_time_series_from_file
                       (START_PATH + "\\" + skill_level + "Data" + "\\" + dir + "\\" + filename)
                       )
    return dataset


# ------- Testing -------- #

# print("Reading in a single time series... ")
# example_time_series = get_single_time_series_from_file(
#     START_PATH + r"\NoviceData\1_B-20170602-102042\NeedleTipToReference-Sequence.mha")
# print("Printing out time series...")
# print(example_time_series, end="\n\n")
#
# print("Reading in all novice time series for needle tip to reference data...")
# dataset = get_subset_of_dataset("Novice", "NeedleTipToReference-Sequence.mha")
# print("Size of dataset: ", len(dataset))
# print("Printing out a randomly selected time series from dataset:")
# print(dataset[random.randint(0, len(dataset) - 1)], end="\n\n")
