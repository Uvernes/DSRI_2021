import random
import math
import numpy as np
from tslearn.barycenters import *
from tslearn.metrics import cdist_dtw
from tslearn.utils import to_time_series_dataset


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


def smote_based_weighted_dba(X, N, k=None, distances=None, multivariate=True):

    """
    Params
    ------

     X : Dataset of time series (uni or multi-variate), all belonging to the same class.
        -Represented as a multi-dim list: outer list stores all the time series, inner lists represent individual time
         series

     N : (int)N/100 is the number of synthetic samples to create per time series in X (i.e amount of SMOTE-ing).
         If N < 100, random subset of X is randomly selected, and 1 synthetic example is created for each time series
         in this subset, where size of subset is floor(N/100 * len(X))

     k: Number of nearest neighbours to check

     distances: Matrix (2D list storing pairwise DTW distances between all time series in X.
            - Entry i,j is DTW(X[i], X[j]) . Note that matrix is symmetric, 0's along main diagonal
            - If not provided, this matrix is computed

     Multivariate: True or false param. Multi and uni-variate require slightly different formatting, so accounted
                  for by this param
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
            synthetic = dtw_barycenter_averaging_petitjean([X[i], X[nn_chosen]], size, np.array(X[i]),
                                                           30, 1e-5, np.array(weights))
            if multivariate:
                synthetic_samples.append(synthetic.tolist())
            else:
                synthetic_samples.append(synthetic.flatten().tolist())

    return synthetic_samples


def smote_based_weighted_dba_specified_amount(X, synthetic_amount, k=None, distances=None, multivariate=True):

    """
    Params
    ------

     X : Dataset of time series (uni or multi-variate), all belonging to the same class.
        -Represented as a multi-dim list: outer list stores all the time series, inner lists represent individual time
         series

     k: Number of nearest neighbours to check

     synthetic_amount - Amount of synthetic samples to create

     distances: Matrix (2D list storing pairwise DTW distances between all time series in X.
            - Entry i,j is DTW(X[i], X[j]) . Note that matrix is symmetric, 0's along main diagonal
            - If not provided, this matrix is computed

     Multivariate: True or false param. Multi and uni-variate require slightly different formatting, so accounted
                  for by this param
     Output
     ------

      2D list storing (size_specified) synthetic time series

    """
    synthetic_samples = []

    if synthetic_amount <= 0:
        return synthetic_samples
    if len(X) == 0:
        print("\n EMPTY DATASET, returning without applying any smote_based_wDBA...\n")
        return synthetic_samples
    if len(X) == 1:
        print("\nONLY 1 DATUM. Not enough to perform smote_based_wDBA. Only duplication will occur...\n")
        while len(synthetic_samples) < synthetic_amount:
            synthetic_samples.append(X[0])
        return synthetic_samples
    if distances is None:
        distances = cdist_dtw(X)
    if k is None:
        k = math.floor(math.sqrt(len(X)))
    # k must be <= len(X) - 1
    if k >= len(X):
        k = len(X) - 1

    # indices order is random to ensure the original samples selected to create synthetic ones are random
    indices_order = list(range(0, len(X)))
    random.shuffle(indices_order)
    while True:
        for i in indices_order:
            nearest_neighbours = find_nearest_neighbours(distances, i, k)
            nn_chosen = nearest_neighbours[random.randint(0, k-1)]
            # Size of synthetic time series is uniformly random between size of two series being averaged
            size = random.randint(min(len(X[i]), len(X[nn_chosen])), max(len(X[i]), len(X[nn_chosen])))
            w_1 = random.random()
            weights = [w_1, 1 - w_1]
            synthetic = dtw_barycenter_averaging_petitjean([X[i], X[nn_chosen]], size, np.array(X[i]),
                                                           30, 1e-5, np.array(weights))
            if multivariate:
                synthetic_samples.append(synthetic.tolist())
            else:
                synthetic_samples.append(synthetic.flatten().tolist())
            if len(synthetic_samples) == synthetic_amount:
                return synthetic_samples


# ----------- Quick testing -------------- #

# print("Uni-variate testing")
# print("-------------------")
# X = [[1, 1, 3], [1, 2, 1], [1, 3, 3], [1, 1], [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1]]
# print("Original uni-variate dataset:")
# print(X)
# synthetic_samples = smote_based_weighted_dba(X, 100, 2, multivariate=False)
#
# print("Synthetic uni-variate time series:")
# for i in synthetic_samples:
#     print(i)

"""
Sample output:

>> Uni-variate testing
>> -------------------
>> Original uni-variate dataset:
>> [[1, 1, 3], [1, 2, 1], [1, 3, 3], [1, 1], [1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1]]
>> Synthetic uni-variate time series:
>> [1.0, 1.0, 2.4133840268603874]
>> [1.0, 2.0, 1.0]
>> [1.0, 3.0, 3.0]
>> [1.0, 2.8892560974516024]
>> [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
"""

# print("\nMultivariate testing")
# print("------------------------")
# X = [ [[1, 4], [2, 5], [3 , 6]], [[2, 5], [3, 6]] ]
# print("Original dataset:")
# print(X)
# print("\nSynthetic sets achieved from wDBA, SMOTE wDBA, respectively:")
# print(dtw_barycenter_averaging_petitjean(X).tolist())
# print(smote_based_weighted_dba_specified_amount([X[0]], 3, 3))

"""
Sample output:

>> Multivariate testing
>> ------------------------
>> Original dataset:
>> [[[1, 4], [2, 5], [3, 6]], [[2, 5], [3, 6]]]
>> Synthetic sets achieved from wDBA, SMOTE wDBA, respectively:
>> [[1.5 4.5]
>>  [2.  5. ]
>>  [3.  6. ]]
>> [[[1.1286235309866437, 4.128623530986644], [2.0, 5.0], [3.0, 6.0]], [[1.669842991153308, 4.669842991153308], [3.0, 6.0]]]
"""