import random

import numpy as np
from utility.nearest_rotation_matrix import exact_nearest_rotation_matrix
from tslearn.barycenters import dtw_barycenter_averaging_petitjean
import sys

"""
This method converts the given 1D list into a transformation matrix (2D numpy array. Note that R matrix may not be orthogonal)
List entered should be of the following form : [R_11, R_12, R_13, t_1, ... , R_31, R_32, R_33, t_3]
"""
def list_to_matrix(t_list): return np.array(t_list + [0, 0, 0, 1]).reshape((4, 4))


"""
This method converts a transformation matrix (2D numpy array) into a 1D list representation (last row ignored)
"""
def matrix_to_list(matrix): return matrix.flatten()[0:12].tolist()


"""
This method takes in a list representation of a transformation matrix and sets the R matrix portion to the proper,
nearest rotation matrix.

Note: Returns a new list (rather than modifying in-place)
"""


def fix_rotation_matrix(t_list):

    t_matrix = list_to_matrix(t_list)
    # print("\nMatrix before:")
    # np.savetxt(sys.stdout, t_matrix, '%.5f')
    # print("Output from function:")
    # np.savetxt(sys.stdout, exact_nearest_rotation_matrix(t_matrix[0:3, 0:3]), '%.5f')
    t_matrix[0:3, 0:3] = exact_nearest_rotation_matrix(t_matrix[0:3, 0:3])
    # print("Matrix after:")
    # np.savetxt(sys.stdout, t_matrix, '%.5f')
    # print("tranpose of R:")
    # np.savetxt(sys.stdout, t_matrix[0:3,0:3].transpose(), '%.5f')
    # print("Check product:")
    # np.savetxt(sys.stdout, np.dot(t_matrix[0:3,0:3], t_matrix[0:3,0:3].transpose()), '%.5f')
    return matrix_to_list(t_matrix)


"""
This method takes in a list representation of a time series and sets all of its rotation matrices to their proper, 
nearest rotation matrices.
Note: No return necessary as input is simply mutated
"""


def fix_rotation_matrices(time_series):

    for i in range(len(time_series)):
        time_series[i] = fix_rotation_matrix(time_series[i])


def convert_all_time_series_to_same_length_using_wdba(dataset, fixed_length):

    for skill_level in dataset:
        for surgery_type in dataset[skill_level]:
            for i in range(len(dataset[skill_level][surgery_type])):
                converted_time_series = \
                     dtw_barycenter_averaging_petitjean([dataset[skill_level][surgery_type][i]],
                                                        barycenter_size=fixed_length).tolist()
                fix_rotation_matrices(converted_time_series)
                dataset[skill_level][surgery_type][i] = converted_time_series


def zero_pad_single_time_series(t_series, fixed_length, position):
    t_series = t_series.tolist()
    for i in range(len(t_series)):
        while len(t_series[i]) < fixed_length:
            if position == "pre":
                t_series[i].insert(0, 0)
            else:
                t_series[i].append(0)

    return np.array(t_series)


def zero_pad_time_series(dataset, fixed_length, position="pre"):
    """
    This function zero-pads all the time series in a given dataset to some fixed-length
    :param dataset - dictionary storing time series
    :param fixed_length - length to pad time series to
    :param position: If 'pre', zeros added to start of time series. If 'post', zeros added to end of time series
    """
    if position != "pre" and position != "post":
        return

    for skill_level in dataset:
        for surgery_type in dataset[skill_level]:
            # Go through all time series for a particular skill level and surgery type
            for i in range(len(dataset[skill_level][surgery_type])):
                dataset[skill_level][surgery_type][i] = \
                    zero_pad_single_time_series(dataset[skill_level][surgery_type][i], fixed_length, position)


def change_format_of_single_time_series(time_series):
    """
    This method takes in a time series represented as a 2D list with inner lists, one inner list for each timestamp.
    It then changes it into an a 2D np array with 12 inner lists, one corresp. to each entry in the transformation matrix.
    (see more details in documentation below).
    """
    # Number of variables measured by transformations (does not include bottom row of matrices)
    num_variables = len(time_series[0])
    prepared = []
    for _ in range(num_variables):
        prepared.append([])
    for transformation in time_series:
        for i in range(num_variables):
            prepared[i].append(transformation[i])

    return np.array(prepared)


def change_format_of_time_series(dataset):
    """
    This function goes through all of the time series in the dataset provided and changes them into the form
    required by model.fit()
    Specifically:
    -Previously time series were 2D lists with inner lists of length 12, and the number of inner lists was equal to
     the number of timestamps
    -Now, time series are 2D np arrays, with inner np arrays having a length equal to the number of timestamps.
    -There are 12 np arrays, each corresponding to an entry in the translation matrix observed across time
     (we ignore the bottom row of matrices)
    :param dataset - dictionary storing time series data, split into experts and novices, and then into IP and OOP
    :return
    """
    for skill_level in dataset:
        for surgery_type in dataset[skill_level]:
            # Go through all time series for a particular skill level and surgery type
            for i in range(len(dataset[skill_level][surgery_type])):
                dataset[skill_level][surgery_type][i] = \
                    change_format_of_single_time_series(dataset[skill_level][surgery_type][i])

    return dataset


# OLD - NO LONGER USED
def prepare_sets_for_model(sets):

    """
    This function goes through all of the time series in all of the sets provided, and changes them into the form
    required to pass them into model_related.utils.prepare_folds(), which performs window slicing

    Specifically:
    -Previously time series were 2D lists with inner lists of length 12, and the number of inner lists was equal to
     the number of timestamps
    -Now, time series are 2D np arrays, with inner np arrays having a length equal to the number of timestamps.
    - There are 12 np arrays, each correspond to an entry in the translation matrix observed across time.
      (we ignore the bottom row)

    Parameters - Dictionary storing all training, validation, and test set data respectively.
    Returns - Three sets, with now all the time series in the proper form. Sets are tuples with 2 elements
               (corresponding to the two skill levels, and elements are lists storing multiple time series)
    """
    # Extracting 3 sets, no longer stored as dictionaries, and not split into IP and OOP surgeries. Still split
    # into novices and experts. Each set stored as a list with 2 elements, one corresponding to each skill level
    train_set, val_set, test_set = [[], []], [[], []], [[], []]
    new_sets = [train_set, val_set, test_set]
    set_names = ["train", "val", "test"]
    for i in range(len(set_names)):
        new_sets[i][0] = sets[set_names[i]]["Novice"]["IP"] + sets[set_names[i]]["Novice"]["OOP"]
        new_sets[i][1] = sets[set_names[i]]["Expert"]["IP"] + sets[set_names[i]]["Expert"]["OOP"]

    num_skill_levels = len(train_set)

    train_set_prepared = [[] for _ in range(num_skill_levels)]
    val_set_prepared = [[] for _ in range(num_skill_levels)]
    test_set_prepared = [[] for _ in range(num_skill_levels)]

    original_sets = (train_set, val_set, test_set)
    prepared_sets = [train_set_prepared, val_set_prepared, test_set_prepared]

    # Goes through all the time series in all the sets, and prepares them. Prepared time series are stored
    # in prepared_sets
    for set_index in range(len(original_sets)):
        cur_set = original_sets[set_index]
        for skill_index in range(num_skill_levels):
            for t_series in cur_set[skill_index]:
                prepared_sets[set_index][skill_index].append(change_format_of_single_time_series(t_series))
        prepared_sets[set_index] = tuple(prepared_sets[set_index])

    return tuple(prepared_sets)


def participants_storage_to_dictionary(storage):
    """
    This function takes in a ParticipantStorage object, and adds all of its time series into a multi-level dictionary.
    The dictionary splits data into novices and experts, and then into IP and OOP surgeries.
    e.g dataset["Novices"]["OOP"] returns a list of all novice OOP time series data
    Note: Dictionary does not keep track of participant names (so conversion cannot be reversed)
    """
    dataset = dict(
        Novices=dict(IP=[], OOP=[]),
        Experts=dict(IP=[], OOP=[])
    )
    for participant in storage.participants.values():
        for surgery in participant.surgeries:
            dataset[surgery.skill_level + "s"][surgery.surgery_type].append(surgery.time_series)

    return dataset


# TESTED - works well
def join_dataset_dictionaries(d1, d2, shuffle=True):

    joined_dataset = dict(
        Novices=dict(IP=[], OOP=[]),
        Experts=dict(IP=[], OOP=[])
    )
    for skill_level in d1:
        for surgery_type in d1[skill_level]:
            for t_series in d1[skill_level][surgery_type]:
                joined_dataset[skill_level][surgery_type].append(t_series)
            for t_series in d2[skill_level][surgery_type]:
                joined_dataset[skill_level][surgery_type].append(t_series)

    if shuffle:
        for skill_level in d1:
            for surgery_type in d1[skill_level]:
                random.shuffle(d1[skill_level][surgery_type])
            for surgery_type in d2[skill_level]:
                random.shuffle(d2[skill_level][surgery_type])

    return joined_dataset

# ------------- Testing ----------------

# t_list = [0.1959, -0.3765, -0.7937, 5, 0.7458, 0.5011, -0.0536, 10, 0.4643, -0.6459, 0.4210, 5]
# t_matrix = list_to_matrix(t_list)
# t_list = matrix_to_list(t_matrix)
# print(t_list)
# print(t_matrix)
# print("t_matrix after fixing R:")
# print(list_to_matrix(fix_rotation_matrix(t_list)))

# print("\n Fixing rotation matrices of some dataset: ")
# dataset = [[0.1959, -0.3765, -0.7937, 5, 0.7458, 0.5011, -0.0536, 10, 0.4643, -0.6459, 0.4210, 5],
#            [0.1959, -0.3765, -0.7937, 5, 0.7458, 0.5011, -0.0536, 10, 0.4643, -0.6459, 0.4210, 5]]
# fix_rotation_matrices(dataset)
#
# print(dataset)

# Testing preparation for model methods
# t_series = [[-0.507056, -0.861912, 0.000902661, -56.4653, 0.384337, -0.225166, 0.895313, -111.779, -0.771478, 0.454321, 0.445437, -8.28349],
#             [-0.500165, -0.865915, 0.00509711, -56.3547, 0.399925, -0.225773, 0.888305, -111.03, -0.768046, 0.446338, 0.459225, -8.18197],
#             [-0.487693, -0.872991, 0.0065299, -56.1794, 0.41034, -0.22262, 0.884343, -110.421, -0.770569, 0.433967, 0.466794, -7.73815]]
#
# t_series_2 = [[-0.507056, -0.861912, 0.000902661, -56.4653, 0.384337, -0.225166, 0.895313, -111.779, -0.771478, 0.454321, 0.445437, -8.28349],
#              [-0.500165, -0.865915, 0.00509711, -56.3547, 0.399925, -0.225773, 0.888305, -111.03, -0.768046, 0.446338, 0.459225, -8.18197]]
#
# print(t_series, "\n")
# print(prepare_time_series_for_model(t_series))
# print()

# train_set = ([t_series, t_series_2], [t_series, t_series])
# val_set = ([t_series, t_series], [t_series, t_series])
# test_set = ([t_series, t_series], [t_series, t_series])
# prepared_sets = prepare_sets_for_model(train_set, val_set, test_set)
# print("Prepared, novice training set:")
# print(prepared_sets[0][0])
# print("Type:", type(prepared_sets[0][0]))
# print("Prepared novice training set, first time series:")
# print(prepared_sets[0][0][0])
# print("Type:", type(prepared_sets[0][0][0]))

