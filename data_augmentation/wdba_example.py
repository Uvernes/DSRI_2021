import random
from tslearn.barycenters import dtw_barycenter_averaging_petitjean
from tslearn.metrics import cdist_dtw
import numpy as np

# ------ Example of applying wDBA to a time series dataset ------- #

# Time series - 3 dimensions each
# of the form: [ [x_1 at timestamp_1, x_2 at timestamp_1, x_3 at timestamp_1]
#                                        , ... ,
#                [x_1 at timestamp_last, x_2 at timestamp_last, x_3 at timestamp_last] ]

t_1 = [[1, 10, 25], [2, 11, 23], [3, 12, 22]]  # 3 timestamps
t_2 = [[3, 15, 35], [5, 17, 29], [6, 20, 28], [7, 22, 21]]
t_3 = [[2, 19, 30], [4, 20, 24], [5, 22, 20]]

dataset = [t_1, t_2, t_3]
print("Dataset: ")
for t_series in dataset:
    print(dataset)

print("\nDistances between all pairwise time series:")
print(cdist_dtw(dataset))

print("\nApplying wDBA to dataset and printing result:")
averaging_weights = np.array([1, 1, 1])
size = random.randint(2, 4)
synthetic = dtw_barycenter_averaging_petitjean(dataset, barycenter_size=size, weights=averaging_weights)
synthetic = synthetic.tolist()
print(synthetic)

# --------Negative weights example --------------- #
# Look into neg. weights later!!

print("Applying wDBA where we pass in negative weights: ")
t_1 = [5, 5, 7, 7, 5, 5]
t_2 = [1, 1, 2, 2, 1, 1]
# t_3 = [2,4,8,9,3,3]
dataset = [t_2, t_1]
averaging_weights1 = np.array([1.25, -0.25])
synthetic = dtw_barycenter_averaging_petitjean(dataset, barycenter_size=6, weights=averaging_weights1)
synthetic = synthetic.flatten().tolist()
print(synthetic)
