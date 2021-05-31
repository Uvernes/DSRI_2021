import numpy as np
import math

"""
Input
-----
R - 3 by 3 matrix for which we want to find its nearest rotation matrix

Output
------
Nearest rotation matrix

"""


def exact_nearest_rotation_matrix(R):

    A = np.dot(R.transpose(), R)
    m = np.trace(A) / 3
    Q = A - m * np.identity(3)
    q = np.linalg.det(Q) / 2
    p = np.sum(tMatrix * 2) / 6
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

    b0 = (a2*a1**2 -a0*(a2**2 +a1))/dem
    b1 = (a0+a2*(a2**2 - 2*a1))/dem
    b2 = a2/dem

    U = b2*A*A -b1*A +b0*np.eye(3)

    return np.dot(R, U)


# --------------- Testing -----------------------#
tMatrix = np.array([[0.1959, -0.3765, -0.7937],
                    [0.7458, 0.5011, -0.0536],
                    [0.4643, -0.6459, 0.4210]])

closest = exact_nearest_rotation_matrix(tMatrix)
print(closest)
# Matrix multiplication below is eye(3), as expected
print(np.dot(closest, closest.transpose()))
