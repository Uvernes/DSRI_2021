import numpy as np
from nearest_rotation_matrix import exact_nearest_rotation_matrix
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


# ------------- Testing ----------------

# #t_list = [0.2, 0.3, 0.5, 2, 0.23, 0.32, 0.56, 7, 0.24, 0.65, 0.23, 8] -- This gives bizarre results for some reason
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

