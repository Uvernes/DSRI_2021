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
from model_related.classes import *
from model_related.model_methods import build_compile_and_fit_model
from utility.data_extraction_and_storage_methods import load_dataset, SKILL_LEVELS, SURGERY_TYPES
from utility.conversion_methods import *
from utility.cross_validation_methods import nested_cv, join_folds
from statistics import mean, stdev
from tabulate import tabulate

from sklearn.metrics import recall_score, confusion_matrix, f1_score
from matplotlib import pyplot as plt

DATASET_PATH = r"C:\Users\uvern\Dropbox\My PC (LAPTOP-554U8A6N)\Documents\DSRI\Data\usneedle_data\SplitManually_Score20_OnlyBF"
SEQUENCE_TYPE = "NeedleTipToReference"
TIME_SERIES_LENGTH_FOR_MODEL = 2200  # Average time series length is ~258.61 , longest is 2191
SLICE_WINDOW = 70  # originally 70
RESULTS_FILE = "results_5_accuracy.txt"

# Used k_outer = 5 , k_inner = 3
# NOTE: FIX BINARY CROSSENTROPY LOSS ISSUE <- THIS NEEDS TO BE MINIMIZED, NOT MAXIMIZED
SELECTED_PERFORMANCE_MEASURE = "Binary accuracy"
K_OUTER = 5
K_INNER = 3  # or 4 (so val and test set ~same size)


# 2 * 2 * 3 * 3 * 3 = 108 combinations
# Note: Dictionaries in Python 3.7+ store keys in insertion order. This fact is used
HYPER_PARAMETERS_GRID = {
    "kernel-size":      [3, 5],                                # originally 5 (7, 10?)
    "filters":          [16, 32],                              # originally 64
    "epochs":           [100, 200, 300],                       # originally 300 (don't tune, use callbacks?)
    "batch-size":       [32],                                  # originally 32
    "dropout-rate":     [0.0, 0.2, 0.5],                       # originally 0.5
    "learning-rate":    [0.0001, 0.001, 0.01],                 # originally 0.0001
    "regularizer":      [0.05]                                 # originally 0.05
}

# HYPER_PARAMETERS_GRID = {
#     "kernel-size":      [3],                                # originally 5
#     "filters":          [32],                              # originally 64
#     "epochs":           [1],                               # originally 300
#     "batch-size":       [32],                              # originally 32
#     "dropout-rate":     [0],                               # originally 0.5
#     "learning-rate":    [0.01],                            # originally 0.0001
#     "regularizer":      [0.05]                             # originally 0.05
# }


def main():

    tic = time.perf_counter()

    # Defines data augmentation to perform in inner / outer training sets
    # data_augmentation = DataAugmentationController(
    #     instructions=[
    #         BalanceDataset(technique=SmoteBasedWDBA(), augment_synthetic=False),
    #         # IncreaseDatasetProportionally(technique=Jittering(), increase_factor=2, augment_synthetic=False),
    #     ]
    # )

    dataset = load_dataset(DATASET_PATH, SEQUENCE_TYPE)
    outer_folds, all_train_results, all_test_results, optimal_configurations = \
        nested_cv(dataset, K_OUTER, K_INNER, HYPER_PARAMETERS_GRID, TIME_SERIES_LENGTH_FOR_MODEL,
                  SELECTED_PERFORMANCE_MEASURE)

    toc = time.perf_counter()

    # Print test results to both the terminal and the specified file
    sys.stdout = Transcript(RESULTS_FILE)

    print("\n=> Results of running nested cv")
    print("   ----------------------------")

    print("\nHours elapsed: %.2f" % (((toc - tic) / 60) / 60))

    print("\nDataset: usneedle_data\\SplitManually_Score20_OnlyBF")

    print("\nHYPER_PARAMETERS_GRID =")
    print(HYPER_PARAMETERS_GRID)

    print("\nSelected performance measure for cross-validation:", SELECTED_PERFORMANCE_MEASURE)
    print("\nTime series length for model:", TIME_SERIES_LENGTH_FOR_MODEL)
    print("\nk-outer:", K_OUTER)
    print("\nk-inner:", K_INNER, "\n\n")

    # Print mean of results
    lists_to_print = []
    for performance_measure in all_test_results:
        lists_to_print.append([performance_measure, mean(all_train_results[performance_measure]),
                               mean(all_test_results[performance_measure])])
    print(tabulate(tabular_data=lists_to_print,
                   headers=["Performance measure", "Training - mean", "Tests - mean"]))

    # Print std of results
    print("\n")
    lists_to_print = []
    for performance_measure in all_test_results:
        lists_to_print.append([performance_measure, stdev(all_train_results[performance_measure]),
                               stdev(all_test_results[performance_measure])])
    print(tabulate(tabular_data=lists_to_print,
                   headers=["Performance measure", "Training - stdev", "Tests - stdev"]))

    print("\n\nAll training results")
    print("--------------------\n")
    lists_to_print = []
    for i in range(len(optimal_configurations)):
        lists_to_print.append(["Set #" + str(i+1)])
        for performance_measure in all_train_results:
            lists_to_print[i].append(all_train_results[performance_measure][i])

    print(tabulate(tabular_data=lists_to_print, headers=["Training set"]+list(all_test_results.keys())))

    print("\n\nAll test results")
    print("----------------\n")
    lists_to_print = []
    for i in range(len(optimal_configurations)):
        lists_to_print.append(["Set #" + str(i+1)])
        for performance_measure in all_test_results:
            lists_to_print[i].append(all_test_results[performance_measure][i])

    print(tabulate(tabular_data=lists_to_print, headers=["Test"]+list(all_test_results.keys())))

    print("\n\nOptimal configurations")
    print("----------------------\n")
    lists_to_print = []
    for i in range(len(optimal_configurations)):
        lists_to_print.append(["Set #" + str(i+1)] + list(optimal_configurations[i]))
    print(tabulate(tabular_data=lists_to_print, headers=["Training/test set"]+list(HYPER_PARAMETERS_GRID.keys())))

    print("\n\nInformation regarding how the data was split into " + str(K_OUTER) + " outer folds:\n\n")

    lists_to_print = [["Dataset", dataset.novices_to_experts_ratio(), dataset.novices_IP_to_OOP_ratio(),
                       dataset.novices_to_experts_ratio()]]
    for i in range(len(outer_folds)):
        training_set = join_folds(outer_folds[0:i] + outer_folds[i+1:])
        lists_to_print.append(["Training set #" + str(i+1), training_set.novices_to_experts_ratio(),
                               training_set.novices_IP_to_OOP_ratio(), training_set.experts_IP_to_OOP_ratio()])
    print(tabulate(tabular_data=lists_to_print, headers=["Sets", "Novices to experts ratio", "Novices IP to OOP ratio",
                                                         "Experts IP to OOP ratio"]))

    print("\n\n")
    lists_to_print = [["Dataset", dataset.novices_to_experts_ratio(), dataset.novices_IP_to_OOP_ratio(),
                       dataset.novices_to_experts_ratio()]]
    for i in range(len(outer_folds)):
        test_set = join_folds(outer_folds[0:i] + outer_folds[i+1:])
        lists_to_print.append(["Test set #" + str(i+1), test_set.novices_to_experts_ratio(),
                               test_set.novices_IP_to_OOP_ratio(), test_set.experts_IP_to_OOP_ratio()])
    print(tabulate(tabular_data=lists_to_print, headers=["Sets", "Novices to experts ratio", "Novices IP to OOP ratio",
                                                         "Experts IP to OOP ratio"]))

    print("\n\n")
    lists_to_print = []
    for i in range(len(outer_folds)):
        training_set = join_folds(outer_folds[0:i] + outer_folds[i+1:])
        lists_to_print.append(["Set #" + str(i+1), training_set.surgeries_stats["novices_IP"],
                               training_set.surgeries_stats["novices_OOP"], training_set.surgeries_stats["experts_IP"],
                               training_set.surgeries_stats["experts_OOP"]])
    print(tabulate(tabular_data=lists_to_print, headers=["Training set", "# Novices IP", "# Novices OOP",
                                                         "# Experts IP", "# Experts OOP"]))

    print("\n\n")
    lists_to_print = []
    for i in range(len(outer_folds)):
        test_set = outer_folds[i]
        lists_to_print.append(["Set #" + str(i+1), test_set.surgeries_stats["novices_IP"],
                               test_set.surgeries_stats["novices_OOP"], test_set.surgeries_stats["experts_IP"],
                               test_set.surgeries_stats["experts_OOP"]])
    print(tabulate(tabular_data=lists_to_print, headers=["Test set", "# Novices IP", "# Novices OOP", "# Experts IP",
                                                         "# Experts OOP"]))

    print("\n\n")

    # Returns print functionality back to normal
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal


main()


