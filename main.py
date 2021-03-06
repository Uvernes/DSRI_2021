import time
import os
import tensorflow as tf
from model_related.classes import *
from utility.data_extraction_and_storage_methods import load_dataset
from utility.conversion_methods import *
from utility.cross_validation_methods import nested_cv, join_folds
from statistics import mean, stdev
from tabulate import tabulate
from numpy.random import seed

DATASET_PATH = r".\usneedle_data\SplitManually_Score20_OnlyBF"
# DATASET_PATH = r"/home/uvernes/projects/def-holden/uvernes/USNeedleClassification/usneedle_data/SplitManually_Score20_OnlyBF"
SEQUENCE_TYPES = ["NeedleTipToReference"]
TIME_SERIES_LENGTH_FOR_MODEL = 2200  # Average time series length is ~258.61 , longest is 2191
SLICE_WINDOW = 70  # originally 70
SCORE_MEASURE = "f1-score"
RESULTS_FILE = "f1score_1.txt"

# Used k_outer = 5 , k_inner = 3 or 5
K_OUTER = 2
K_INNER = 2  # or 4 (so val and test set ~same size)

# 3 * 2 * 3 * 2 * 2 = 72 combinations
# Note: Dictionaries in Python 3.7+ store keys in insertion order. This fact is used
HYPER_PARAMETERS_GRID = {
    "epochs":           list(range(25, 1001, 25)),            # originally 300 (don't tune, use callbacks?)
    "kernel-size":      [3, 5, 7],                              # originally 5 (7, 10?) ** avoid even numbers
    "filters":          [8, 16],                                # originally 64
    "batch-size":       [32],                                   # originally 32
    "dropout-rate":     [0, 0.5],                               # originally 0.5
    "learning-rate":    [0.0001, 0.0005, 0.001],                # originally 0.0001
    "regularizer":      [0, 0.05]                               # originally 0.05
}

HYPER_PARAMETERS_GRID = {
    "epochs":           list(range(25, 301, 25)),         # list(range(100, 501, 50))
    "kernel-size":      [7],                                # [3, 5, 10]
    "filters":          [16],                                # [8, 16] or [64]
    "batch-size":       [32],                                # originally 32
    "dropout-rate":     [0.5],                               # originally 0.5
    "learning-rate":    [0.0001],                            # [0.01, 0.02]
    "regularizer":      [0.05]                               # originally 0.05
}


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # comment out to use GPU
    # seed(1)

    tic = time.perf_counter()

    # Defines data augmentation to perform in inner training sets
    data_augmentation = DataAugmentationController(
        instructions=[
            IncreaseDatasetBalanced(technique=WindowSlicing(window_slice_prop=0.9), increase_factor=2, augment_synthetic=False),
        ]
    )

    dataset = load_dataset(DATASET_PATH, SEQUENCE_TYPES)
    outer_folds, all_best_train_results, all_best_val_results, all_test_results, optimal_configurations \
        = nested_cv(dataset, K_OUTER, K_INNER, HYPER_PARAMETERS_GRID, TIME_SERIES_LENGTH_FOR_MODEL, SCORE_MEASURE,
                    data_augmentation)

    toc = time.perf_counter()

    # Print test results to both the terminal and the specified file
    # sys.stdout = Transcript(RESULTS_FILE)

    print("\n=> Results of running nested cv")
    print("   ----------------------------")

    print("\nHours elapsed: %.2f" % (((toc - tic) / 60) / 60))

    print("\nDataset: usneedle_data\\SplitManually_Score20_OnlyBF")

    print("\nHYPER_PARAMETERS_GRID =")
    print(HYPER_PARAMETERS_GRID)

    print("\nTime series length for model:", TIME_SERIES_LENGTH_FOR_MODEL)
    print("\nScore measure:", SCORE_MEASURE)
    print("\nk-outer:", K_OUTER)
    print("\nk-inner:", K_INNER, "\n\n")

    # Print mean of results
    lists_to_print = []
    for performance_measure in all_test_results:
        lists_to_print.append([performance_measure, mean(all_best_train_results[performance_measure]),
                               mean(all_best_val_results[performance_measure]), mean(all_test_results[performance_measure])])
    print(tabulate(tabular_data=lists_to_print,
                   headers=["Performance measure", "All training results - mean", "All validation results - mean",
                            "Tests - mean"]))

    # Print stdev of results
    print("\n")
    lists_to_print = []
    for performance_measure in all_test_results:
        lists_to_print.append([performance_measure, stdev(all_best_train_results[performance_measure]),
                               stdev(all_best_val_results[performance_measure]), stdev(all_test_results[performance_measure])])
    print(tabulate(tabular_data=lists_to_print,
                   headers=["Performance measure", "All training results - stdev", "All validation results - stdev",
                            "Tests - stdev"]))

    print("\n\nTraining results for best models found")
    print("----------------------------------------------------------\n")
    lists_to_print = []
    for i in range(len(optimal_configurations)):
        lists_to_print.append(["Result #" + str(i+1)])
        for performance_measure in all_best_train_results:
            lists_to_print[i].append(all_best_train_results[performance_measure][i])
    print(tabulate(tabular_data=lists_to_print, headers=["Result"]+list(all_test_results.keys())))

    print("\n\nValidation results for best models found")
    print("----------------------------------------------------------\n")
    lists_to_print = []
    for i in range(len(optimal_configurations)):
        lists_to_print.append(["Result #" + str(i+1)])
        for performance_measure in all_best_val_results:
            lists_to_print[i].append(all_best_val_results[performance_measure][i])
    print(tabulate(tabular_data=lists_to_print, headers=["Result"]+list(all_test_results.keys())))

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
    # sys.stdout.logfile.close()
    # sys.stdout = sys.stdout.terminal


main()


