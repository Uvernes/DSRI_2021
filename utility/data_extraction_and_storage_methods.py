import os
import shutil
import random

SKILL_LEVELS = ["Novice", "Expert"]
SEQUENCE_TYPES = ["ImageToReference", "NeedleTipToImage", "NeedleTipToReference", "NeedleToProbe",
                  "NeedleToReference", "ProbeToReference"]
SEQUENCES_WITH_SEQ_EXTENSION = SEQUENCE_TYPES[-3:]


def get_single_time_series_from_file(filename):

    """
    This method extracts the time series data from a single MHA file and stores it in a 2D list.
    Note: Last
    Parameters
    ----------
    filename - mha file to extract time series from. Currently, NeedleAndProbeToReference files are not supported

    Returns
    ------
    1) 2D List corresponding to time series of 12 variables (9 from R matrix, 3 from translation vector).
       Inner lists are of the following form:
        [R_11, R_12, R_13, t_1, ... , R_31, R_32, R_33, t_3]

    2) List of timestamps for the given time series extracted.
       i.e [time_1, ..., time_last]
    """

    time_series = []
    timestamps = []
    f = open(filename)
    prev_transform_ok = True
    for line in f:

        if ("ReferenceTransform" not in line) and ("SequenceTransform" not in line) and ("Timestamp" not in line):
            continue

        split_line = line.replace("=", " ").split()
        if "Status" in line:
            if split_line[1] != "OK":
                time_series.pop()
                prev_transform_ok = False
            continue

        if "Timestamp" in line:
            if not prev_transform_ok:
                prev_transform_ok = True
            else:
                timestamps.append(float(split_line[1]))
            continue

        temp_list = []
        for i in range(1, 13):
            temp_list.append(float(split_line[i]))
        time_series.append(temp_list)
    f.close()
    return time_series, timestamps


def get_filename(sequence_type):

    """
    This method returns the filename given the specified sequence type
    """

    if sequence_type in SEQUENCES_WITH_SEQ_EXTENSION:
        return sequence_type + "-Sequence.seq.mha"
    return sequence_type + "-Sequence.mha"


def get_subset_of_dataset(directory, skill_level, sequence_type):

    """
    This method extracts all time series data available for a particular skill level (e.g novice), including the timestamps.
    **Later extensions:
      1) Optional parameter to extract only IP or OOP data
      2) Optional parameter to exclude final trials for novices (at this point they may no longer be novices)

    Parameters
    ----------
    directory - Path to where all data is located
    skill_level - Particular skill level to extract all available time series data for

    sequence_type - Type of sequence being considered (e.g NeedleTipToReference). Do not include extensions (e.g .seq.mha)
                    or the '-Sequence' term

    returns
    ------
    1) Time series dataset (subset of all data) for a particular skill level
    2) Set of all timestamps.
        i.e a 2D list where the i'th element is the list of timestamps for the i'th sample in the dataset

    Note: The two returns above are parallel lists
    """

    dataset = []
    timestamps_set = []
    if (skill_level not in SKILL_LEVELS) or (sequence_type not in SEQUENCE_TYPES):
        return dataset, timestamps_set

    filename = get_filename(sequence_type)

    directories = os.listdir(directory + "\\" + skill_level + "Data")
    for dir in directories:
        if "sqbr" in dir:
            continue
        # print(dir)
        # print(START_PATH + "\\" + skill_level + "Data" + "\\" + dir + "\\" + sequence_type)
        t_series, timestamps = get_single_time_series_from_file(directory + "\\" + skill_level + "Data" + "\\" + dir + "\\" + filename)
        dataset.append(t_series)
        timestamps_set.append(timestamps)
    return dataset, timestamps_set


def load_dataset(directory, sequence_type):
    """
    Parameters
    ----------
    directory - Path to directory where usneedle_data is located
    sequence_type - Type of sequence being considered (e.g NeedleTipToReference). Do not include extensions (e.g .seq.mha)
                    or the '-Sequence' term
    Returns
    -------
    Two lists with 2 elements each, corresponding to each skill level (novice and expert, respectively).
    -In the first list, each element is a list containing all time series data for a particular skill level.
    Note that all time series are lists (not np arrays)
    -In the second, each element is a list containing all the timestamps for all the time series of a particular
     skill level
    """
    novices, novices_timestamps = get_subset_of_dataset(directory, "Novice", sequence_type)
    experts, expert_timestamps = get_subset_of_dataset(directory, "Expert", sequence_type)

    return [novices, experts], [novices_timestamps, expert_timestamps]


def write_time_series_to_mha_file(time_series, timestamps, sequence_type, path):

    """"
    This method takes in a single time series for some sequence type and its corresponding list of timestamps, and writes
    it to an mha file at the specified path.
    """

    filename = get_filename(sequence_type)
    f = open(path + "\\" + filename, "w")

    # Header info
    f.write((
        "ObjectType = Image\n"
        "NDims = 3\n"
        "AnatomicalOrientation = RAI\n"
        "BinaryData = True\n"
        "CompressedData = False\n"
        "DimSize = 0 0 "
    ))
    f.write(str(len(timestamps)) + "\n")
    f.write((
        "ElementSpacing = 1 1 1\n"
        "Offset = 0 0 0\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        "ElementType = MET_UCHAR\n"
        "Kinds = domain domain list\n"
    ))

    transform_label = sequence_type
    if ".seq" in filename:
        transform_label += "-SequenceTransform"
    else:
        transform_label += "Transform"

    for i in range(len(timestamps)):
        sequence_num_label = "Seq_Frame" + str(i).rjust(4, '0') + "_"
        # print transform
        # Note: transform values are currently not rounded
        cur_transform_values = ' '.join([str(elem) for elem in (time_series[i] + [0, 0, 0, 1])])
        f.write(sequence_num_label + transform_label + " =" + cur_transform_values + " \n")
        # print transform status
        f.write(sequence_num_label + transform_label + "Status = OK\n")
        # print timestamp
        f.write(sequence_num_label + "Timestamp = " + "{:.3f}".format(timestamps[i]) + "\n")

    f.write("ElementDataFile = LOCAL")
    f.close()


def write_dataset_to_mha_files(dataset, timestamps_set, sequence_type, directory_path,
                               clear_existing_synthetic_directories=False,
                               add_to_existing_synthetic_directories=True):

    """
    This method takes in a dataset of time series for some sequence type, and its corresponding timestamps set, and writes
     each time series to an MHA file, within the specified directory path.

    Note 1: MHA files are written within their own directories, labelled of the form i_synthetic , where i refers to the
            i'th synthetic sample.

    Note 2: i_synthetic directories may contain multiple mha files (e.g NeedleTipToReference and ProbeToReference).
            MHA files are written to existing directories if the following conditions are all met:
            1) clear_existing_synthetic_directories is False
            2) add_to_existing_synthetic_directories is True
            3) Number of existing synthetic directories equals the size of the data set passed in
               (o.w doesn't make sense)

            - If 1) is met but not 2) or 3), then new directories are added to the bottom of the existing ones

    Parameters
    -----------
    dataset - Dataset of time series
    timestamps_set - Set of timestamps for the dataset
    sequence_type - Sequence type specifies name of the mha files to be written (e.g NeedleTipToReference)
    directory_path - Path of directory where all synthetic directories will be written (with MHA files inside)
    clear_existing_synthetic_directories - If true, clears all sub-directories inside directory_path
    add_to_existing_synthetic_directories - If true, and conditions of 'Note 2' are satisfied, mha files
                                            are added to existing synthetic directories
    """

    prev_synthetic_directories = [ele for ele in os.listdir(directory_path) if os.path.isdir(directory_path + "\\" + ele)]
    if clear_existing_synthetic_directories:
        add_to_existing_synthetic_directories = False
        for ele in prev_synthetic_directories:
            shutil.rmtree(directory_path + "\\" + ele)
        prev_synthetic_directories = []

    if len(prev_synthetic_directories) != len(dataset):
        add_to_existing_synthetic_directories = False

    # Redefine synthetic_directories to make sure it is in ascending order
    prev_num_synthetic_directories = len(prev_synthetic_directories)
    prev_synthetic_directories = []
    for i in range(prev_num_synthetic_directories):
        prev_synthetic_directories.append(str(i + 1) + "_synthetic")

    if add_to_existing_synthetic_directories:
        for i in range(len(dataset)):
            write_time_series_to_mha_file(dataset[i], timestamps_set[i], sequence_type,
                                          directory_path + "\\" + prev_synthetic_directories[i])
    else:
        for i in range(len(dataset)):
            # Create new synthetic directory if not adding to a pre-existing one
            new_dir = directory_path + "//" + str(len(prev_synthetic_directories) + 1 + i) + "_synthetic"
            os.makedirs(new_dir)
            write_time_series_to_mha_file(dataset[i], timestamps_set[i], sequence_type, new_dir)


# ----------- Testing ------------ #

# # Reading in a single series
# print("Reading in a single time series... ")
# example_time_series, timestamps = get_single_time_series_from_file(
#     START_PATH + r"\NoviceData\1_B-20170602-102042\ProbeToReference-Sequence.seq.mha")
# print("Printing out time series (and timestamps)...")
# # print(example_time_series, end="\n\n")
# for ele in example_time_series:
#     print(ele)
# print(timestamps)
# print("Number of timestamps: ", len(timestamps))
#
# # Reading in entire novice dataset
# print("\nReading in all novice time series for needle tip to reference data...")
# dataset, timestamps_set = get_subset_of_dataset("Novice", "NeedleTipToReference")
# print("Size of dataset: ", len(dataset))
# print("\nPrinting out a randomly selected time series from dataset (and timestamps):")
# index = random.randint(0, len(dataset) - 1)
# print(dataset[index])
# print(timestamps_set[index])
# print("Number of timestamps: ", len(timestamps_set[index]))
#
# # Writing one time series to an MHA file
# print("\nWriting to an mha file using example_time_series...")
# path = START_PATH + r"\SyntheticData\ExampleData"
# write_time_series_to_mha_file(example_time_series, timestamps, "ProbeToReference", path)
#
# # Writing entire dataset to synthetic data directory
# write_dataset_to_mha_files(dataset, timestamps_set, "NeedleTipToReference", path, True, False)



