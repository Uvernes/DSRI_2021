import numpy as np
from utility.nearest_rotation_matrix import exact_nearest_rotation_matrix
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


def prepare_time_series_for_model(time_series):
    """
    This method takes in a time series represented as a 2D list with inner lists, one inner list for each timestamp.
    It then changes it into an a 2D np array with 16 inner lists, one corresp. to each entry in the transformation matrix.
    (see more details in documentation below).
    """
    num_variables = 12   # Number of variables measured by transformations (not including bottom row of matrices)
    prepared = []
    for _ in range(16):
        prepared.append([])
    for transformation in time_series:
        for i in range(16):
            if i < num_variables:
                prepared[i].append(transformation[i])
            elif i < num_variables + 3:
                prepared[i].append(0)
            else:
                prepared[i].append(1)

    return np.array(prepared)


def prepare_sets_for_model(train_set, val_set, test_set):

    """
    This function goes through all of the time series in all of the sets provided, and changes them into the form
    required to pass them into model_related.utils.prepare_folds(), which performs window slicing

    Specifically:
    -Previously time series were 2D lists with inner lists of length 12, and the number of inner lists was equal to
     the number of timestamps
    -Now, time series are 2D np arrays, with inner np arrays having a length equal to the number of timestamps.
    - There are 16 np arrays, each correspond to an entry in the translation matrix observed across time.

    Parameters - training, validation, and test set respectively. Each are tuples with 2 elements
                 (this separates novice and expert data)
    Returns - The three sets, with now all the time series in the proper form. Sets are tuples, and elements are lists
              storing multiple time series
    """
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
                prepared_sets[set_index][skill_index].append(prepare_time_series_for_model(t_series))
        prepared_sets[set_index] = tuple(prepared_sets[set_index])

    return tuple(prepared_sets)

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

