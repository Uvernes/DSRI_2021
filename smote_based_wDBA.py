import random
import math
import numpy
from tslearn.barycenters import dtw_barycenter_averaging_petitjean
from tslearn.metrics import cdist_dtw


def smote_based_weighted_dba(X, N, k=None, distances=None):

    """
    Params
    ------

     X : Dataset of uni-variate time series, all belonging to the same class.
        -Represented as a 2D list: outer list stores the time series, inner lists represent individual time series

     N : (int)N/100 is the number of synthetic samples to create per time series in X (i.e amount of SMOTE-ing).
         If N < 100, random subset of X is randomly selected, and 1 synthetic example is created for each time series
         in this subset, where size of subset is floor(N/100 * len(X))

     k: Number of nearest neighbours to check

     distances: Matrix (2D list storing pairwise DTW distances between all time series in X.
            - Entry i,j is DTW(X[i], X[j]) . Note that matrix is symmetric, 0's along main diagonal
            - If not provided, this matrix is computed

     Output
     ------

      N/100 * size(X) synthetic time series, stored in a 2D list

    """

    if distances is None:
        distances = cdist_dtw(X)
    if k is None:
        k = math.floor(math.sqrt(len(X)))

    selected_indices = list(range(0, len(X)))
    if N < 100:
        random.shuffle(selected_indices)
        selected_indices = selected_indices[0:math.floor(N/100 * len(X))]
        N = 100
    N = int(N/100)

    synthetic_samples = []

    for i in selected_indices:
        nearest_neighbours = find_nearest_neighbours(distances,i,k)
        for j in range(N):
            nn_chosen = nearest_neighbours[random.randint(0, k-1)]
            # Size of synthetic time series is uniformly random between size of two series being averaged
            size = random.randint(min(len(X[i]), len(X[nn_chosen])), max(len(X[i]), len(X[nn_chosen])))
            w_1 = random.random()
            weights = [w_1, 1 - w_1]
            synthetic = dtw_barycenter_averaging_petitjean([X[i], X[nn_chosen]], size, numpy.array(X[i]),
                                                           30, 1e-5, numpy.array(weights))
            synthetic = synthetic.flatten().tolist()
            synthetic_samples.append(synthetic)
    return synthetic_samples


def find_nearest_neighbours(distances, t_index, k):

    """
    Params
    -------

    distances: Matrix (2D list) of pairwise DTW distances between all time series in dataset of interest

    t_index: Index of time series in dataset to find k nearest neighbours for

    k: Number of nearest neighbours to find

    Output
    -------

    1D list of indices in dataset corresponding to the k nearest neighbours, ordered from closest to furthest

    """

    nearest_neighbours = []
    # iterate through row (t_index) of distances
    for col in range(len(distances)):
        if col == t_index:
            continue
        cur_distance = distances[t_index][col]
        # Place index of currently inspected time series into nearest_neighbours if it belongs there
        placed = False
        for i in range(len(nearest_neighbours)):
            if cur_distance < distances[t_index][nearest_neighbours[i]]:
                nearest_neighbours.insert(i, col)
                placed = True
                break
        if (not placed) and (len(nearest_neighbours) < k):
            nearest_neighbours.append(col)
        if len(nearest_neighbours) > k:
            nearest_neighbours.pop()
    return nearest_neighbours


# ----------- Quick testing -------------- #

X = [[1, 1, 3], [1, 2, 1], [1, 3, 3], [1, 1], [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1]]
synthetic_samples = smote_based_weighted_dba(X, 100, 2)

print("Synthetic uni-variate time series:")
for i in synthetic_samples:
    print(i)
    
    
"""
Sample output:

>> Synthetic uni-variate time series:
>> [1.0, 1.0, 3.0]
>> [1.0, 2.0, 1.0]
>> [1.0, 2.691956557219443, 2.3839131144388856]
>> [1.0, 2.4404417811400174]
>> [1.0, 1.0, 1.0, 1.17775962614494, 1.17775962614494, 1.17775962614494, 1.17775962614494, 1.17775962614494, 1.0, 1.0, 1.0, 1.0]

"""
