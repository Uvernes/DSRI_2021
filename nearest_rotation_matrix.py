import numpy as np
import math
import sys

"""
Input
-----
R - 3 by 3 matrix we want to find nearest rotation matrix for

Output
------
Nearest rotation matrix

"""


def exact_nearest_rotation_matrix(R):

    A = np.dot(R.transpose(), R)
    m = np.trace(A) / 3
    Q = A - m * np.identity(3)
    q = np.linalg.det(Q) / 2
    p = np.sum(np.power(Q, 2)) / 6
    sp = math.sqrt(p)

    theta = math.atan2(math.sqrt(abs(p**3 - q**2)), q) / 3

    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    l1 = abs(m + 2*sp*ctheta)
    l2 = abs(m - sp*(ctheta+math.sqrt(3)*stheta))
    l3 = abs(m - sp*(ctheta-math.sqrt(3)*stheta))

    a0 = math.sqrt(l1*l2*l3)
    a1 = math.sqrt(l1*l2)+math.sqrt(l1*l3)+math.sqrt(l3*l2)
    a2 = math.sqrt(l1)+math.sqrt(l2)+math.sqrt(l3)

    dem = a0*(a2*a1-a0)

    b0 = (a2*(a1**2) - a0 * ((a2**2) + a1))/dem
    b1 = (a0+a2*((a2**2) - 2*a1))/dem
    b2 = a2/dem

    U = np.dot(np.dot(b2, A), A) - np.dot(b1, A) + np.dot(b0, np.eye(3))

    return np.dot(R, U)


# --------------- Testing -----------------------#
# tMatrix = np.array([[0.74359, -0.66199, 0.08425],
#                     [0.21169, 0.35517, 0.91015],
#                     [-0.63251, -0.65879, 0.40446]])
#
# closest = exact_nearest_rotation_matrix(tMatrix)
# print("Matrix: ")
# print(tMatrix)
# print("Nearest rotation matrix: ")
# print(closest)
# # Matrix multiplication below is eye(3), as expected
# print("Matrix multiplication:")
# np.savetxt(sys.stdout, np.dot(closest, closest.transpose()), '%.5f')

# print("---------------")
# matrix = np.array([[1, 2, 3], [4, 5, 6]])
# print(matrix)
# print(np.sum(matrix))
# print(np.power(matrix, 2))
# print(np.sum(np.power(matrix, 2)))

