import tslearn
import math
import random
import time
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sn
import itertools
import model_related.classes
import model_related.utils as ut
from data_augmentation.jittering import apply_jittering_to_dataset
from data_augmentation.smote_based_wDBA import smote_based_weighted_dba
from data_augmentation.timestamp_methods import summary_stats_of_timestamp_lengths
from model_related.classes import ProficiencyLabel
from model_related.model_methods import build_compile_and_fit_model
from utility.data_extraction_and_storage_methods import load_dataset, SKILL_LEVELS, SURGERY_TYPES
from utility.conversion_methods import *
from utility.cross_validation_methods import nested_cv
from statistics import mean

from sklearn.metrics import recall_score, confusion_matrix, f1_score
from matplotlib import pyplot as plt

DATASET_PATH = r"C:\Users\uvern\Dropbox\My PC (LAPTOP-554U8A6N)\Documents\DSRI\Data\usneedle_data\SplitManually_Score20_OnlyBF"
SEQUENCE_TYPE = "NeedleTipToReference"
TIME_SERIES_LENGTH_FOR_MODEL = 2200  # Average time series length is ~258.61 , longest is 2191
SLICE_WINDOW = 70  # originally 70

K_OUTER = 5
K_INNER = 3  # or 4 (so val and test set ~same size)

# 2 * 2 * 3 * 3 * 3 = 108 combinations
# Note: Dictionaries in Python 3.7+ store keys in insertion order. This fact is used
HYPER_PARAMETERS_GRID = {
    "kernel-size":      [3, 5],                                # originally 5 (7, 10?)
    "filters":          [16, 32],                              # originally 64 (instead, 16, 32)
    "epochs":           [100, 200, 300],                       # originally 300 (don't tune. Use callbacks)
    "batch-size":       [32],                                  # originally 32
    "dropout-rate":     [0.0, 0.2, 0.5],                       # originally 0.5
    "learning-rate":    [0.0001, 0.001, 0.01],                 # originally 0.0001
    "regularizer":      [0.05]                                 # originally 0.05
}

# HYPER_PARAMETERS_GRID = {
#     "kernel-size":      [5],                                # originally 5
#     "filters":          [64],                              # originally 64
#     "epochs":           [5, 10],                             # originally 300
#     "batch-size":       [32],                              # originally 32
#     "dropout-rate":     [0.2],                             # originally 0.5
#     "learning-rate":    [0.0001],                   # originally 0.0001
#     "regularizer":      [0.05]                             # originally 0.05
# }


def main():

    tic = time.perf_counter()
    dataset = load_dataset(DATASET_PATH, SEQUENCE_TYPE)
    # Test results will later be a dictionary <- so it can store all AUC results, f1-score results, etc.
    # Optimal configurations is parallel to each of the lists in test_results.values()
    test_results, optimal_configurations = nested_cv(dataset, "f1-score", K_OUTER, K_INNER,
                                                     HYPER_PARAMETERS_GRID, TIME_SERIES_LENGTH_FOR_MODEL)

    print("\nTest results (f1-scores): ")
    print(test_results)
    print("\nCorresponding, optimal configurations:")
    print(optimal_configurations)

    print("\nMean f1-score:")
    print(mean(test_results))

    toc = time.perf_counter()
    print("\nHours elapsed: %.2f" % ((toc - tic) / 60 / 60))

    # Writing results to file
    f = open("results.txt", "w")
    f.write("Hours elapsed: %.2f\n\n" % (((toc - tic) / 60) / 60))
    f.write("Mean f1-score: " + str(mean(test_results)) + "\n\n")
    f.write("Test results (f1-scores) and corresponding optimal configurations:\n\n")
    for i in range(len(test_results)):
        f.write("f1-score: " + str(test_results[i]) + "\n")
        f.write("Configuration " + str(optimal_configurations[i]) + "\n\n")

    f.close()


main()


