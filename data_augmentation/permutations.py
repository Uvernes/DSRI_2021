import random
import math


# Unfinished - Probably NOT useful for our data
def permute_single_time_series(time_series, k, prop_of_groups_to_permute):
    """
    This method splits the data into groups of size k, and randomly selects a prop. of these groups to apply in-group
    permuting to

    Parameters
    ----------
    k: Timestamps are grouped into groups of size k (last group may be < k)
    prop_of_groups_to_permute: Float b/w 0 and 1 indicating the prop. of groups to apply in group permuting to
    Return
    -------
    """
    groups_starting_indices = [i * k for i in list(range(0, math.ceil(len(time_series)/k)))]
    print(groups_starting_indices)
    random.shuffle(groups_starting_indices)
    groups_to_permute = groups_starting_indices[0:math.floor(prop_of_groups_to_permute * len(groups_starting_indices))]
    print(groups_to_permute)


permute_single_time_series([1,2,3,4,5,6,7,8,9,10],3, 0.5)
