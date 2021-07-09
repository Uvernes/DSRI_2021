import itertools
import math
import model_related.utils as ut
import tensorflow as tf
from statistics import mean
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score, accuracy_score
from model_related.model_methods import build_compile_and_fit_model
from utility.conversion_methods import *
from utility.k_folds_methods import split_dataset_into_k_folds, join_folds
from copy import copy


VALIDATION_PERFORMANCE_MEASURES = ["Binary crossentropy loss", "Binary accuracy", "AUC", "f1-score",
                                   "Precision", "Recall"]


# Validation performances measures measured in regular_cv
def performance_measures_dict():
    dictionary = dict()
    for performance_measure in VALIDATION_PERFORMANCE_MEASURES:
        dictionary[performance_measure] = []
    return dictionary


def compute_performance_measures(model, set, threshold=0.5, find_optimal_threshold=False):
    performance_results = dict()
    x = set[0]
    y_true = set[1]
    y_pred = model.predict(x).flatten()

    # print("y_true:", y_true)
    # print("\ny_pred:", y_pred)
    # print("\ny_pred_rounded", y_pred_rounded)

    # AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    performance_results["AUC"] = auc(fpr, tpr)

    # Find the best threshold - Youden’s J statistic method
    if find_optimal_threshold:
        J = tpr - fpr
        ix = np.argmax(J)
        threshold = thresholds[ix]

    y_pred_rounded = np.where(y_pred > threshold, 1, 0)

    # Binary crossentropy loss
    performance_results["Binary crossentropy loss"], prev_accuracy = \
        model.evaluate(x=x, y=y_true, verbose=0)

    # Binary accuracy
    performance_results["Binary accuracy"] = accuracy_score(y_true, y_pred_rounded)

    # Precision, recall
    performance_results["Precision"] = precision_score(y_true, y_pred_rounded, zero_division=0)
    performance_results["Recall"] = recall_score(y_true, y_pred_rounded, zero_division=0)

    # f1-score
    performance_results["f1-score"] = f1_score(y_true, y_pred_rounded)

    if find_optimal_threshold:
        return performance_results, threshold
    return performance_results


def get_best_model(all_val_results, all_configurations, all_models):
    """
    This function takes in all of the validation results from grid search and the corresponding configurations,
    and it returns the best model, as well as the corresponding hyper-parameters and index
    :param all_val_results: A list of dictionaries for each configuration. Each dictionary stores all val results
                            for a given configuration
    :param all_configurations: The i'th dictionary in all_val_results corresponds to the results for the i'th
                               configuration in all_configurations
    :return: best model (i.e one with highest score), as well as the corresponding index
    """
    # Dictionary with performance measures as keys. Values are lists storing the average results for a particular performance
    # measure, for each configuration. So values are lists of average results that are parallel to all_configurations
    average_val_results = performance_measures_dict()
    for results in all_val_results:
        for measure in VALIDATION_PERFORMANCE_MEASURES:
            average_val_results[measure].append(mean(results[measure]))

    print("\naverage_val_results, for each configuration:")
    print(average_val_results)

    # print("\nAll configurations:")
    # print(all_configurations)

    # Measures we look at for determining scores. Scores are the squared rooted, sum of these results
    selected_measures = ["Binary accuracy", "AUC", "f1-score"]
    scores = []
    for i in range(len(all_configurations)):
        cur_score = 0
        for performance_measure in selected_measures:
            # Square rooting gives preference to results that have less 'imbalances' between performance measures
            cur_score += math.sqrt(average_val_results[performance_measure][i])
            # Scores are normalized so they fall in the range [0,1] - doesn't affect which hyper-params chosen since
            # scores all multiplied by the same scalar
        scores.append((cur_score / len(selected_measures)))

    print("\nScores:\n", scores, "\n")

    best_hyper_params_index = scores.index(max(scores))

    # From the K_INNER models trained using the best hyper parameters, select the one with the best individual val
    # results.
    candidate_models = all_models[best_hyper_params_index]
    best_model = None
    best_model_index = 0
    best_score = -1

    for i in range(len(candidate_models)):
        # Compute individual score for the current model
        cur_score = 0
        for performance_measure in selected_measures:
            cur_score += math.sqrt(all_val_results[best_hyper_params_index][performance_measure][i])
        if cur_score > best_score:
            best_score = cur_score
            best_model = all_models[best_hyper_params_index][i]
            best_model_index = i

    # Return index of hyper-parameters corresponding to the highest score (or first highest if there are ties)
    return best_model, best_hyper_params_index, best_model_index


# i.e non-nested cv. Computes all performance measures of interest, for each fold, for various # of epochs
def regular_cv(train_set, k, hyper_params, epochs_list, fixed_length_for_time_series, data_augmentation):
    # We have E validation results dictionaries, where E = len(epochs_list). Same for training results
    all_train_results = []
    all_val_results = []

    # Store the (K_INNER) models trained for each hyper parameter dictionary. i.e models is a 2D list, where the
    # inner list at index i stores the (K_INNER) models for the i'th configurations (configs differ by # of epochs)
    models = []

    for _ in epochs_list:
        all_val_results.append(performance_measures_dict())
        all_train_results.append(performance_measures_dict())
        models.append([])

    inner_folds = split_dataset_into_k_folds(train_set, k)
    for i in range(k):

        inner_train_set = join_folds(inner_folds[0:i] + inner_folds[i + 1:])
        inner_val_set = inner_folds[i]

        # print("Inner training set surgeries:", inner_train_set.surgeries_stats["surgeries"])
        # print("Inner validation set surgeries:", inner_val_set.surgeries_stats["surgeries"])

        # Convert sets (ParticipantsStorage objects) into dictionaries of time series.
        # Time series of different sequence types for the same surgery are concatenated at each timestamp
        inner_train_set = participants_storage_to_dictionary(inner_train_set)
        inner_val_set = participants_storage_to_dictionary(inner_val_set)

        # print("Number of novices IP, OOP, experts IP, OOP, respectively in inner training set:")
        # print(len(inner_train_set["Novices"]["OOP"]), "|", len(inner_train_set["Novices"]["IP"]), "|",
        #       len(inner_train_set["Experts"]["IP"]), "|", len(inner_train_set["Experts"]["OOP"]))

        # ------- Apply data augmentation to training set here (potentially to val set too?) ------ #
        if data_augmentation is not None:
            # inner_train_set enhanced
            synthetic_dataset = data_augmentation.execute(inner_train_set)
            inner_train_set = join_dataset_dictionaries(inner_train_set, synthetic_dataset)

        # changes all time series into the format expected by model.fit()
        change_format_of_time_series(inner_train_set)
        change_format_of_time_series(inner_val_set)

        # pad both time series (later, address time series longer than fixed_length)
        # Also, each time series becomes an np array
        zero_pad_time_series(inner_train_set, fixed_length=fixed_length_for_time_series)
        zero_pad_time_series(inner_val_set, fixed_length=fixed_length_for_time_series)

        # LATER - perform window slicing

        # Sets are tuples with 2 elements (np arrays). 1st element stores the data, 2nd stores the labels
        inner_train_set = ut.split_set_into_x_and_y(inner_train_set)
        inner_val_set = ut.split_set_into_x_and_y(inner_val_set)
        model = None
        for j in range(len(epochs_list)):
            if j == 0:
                # function below expects hyper_params to include the number of epochs
                # Note: model is only built once, for the lowest 'number of epochs' value
                hyper_params["epochs"] = epochs_list[j]
                print("Number of epochs trained:", hyper_params["epochs"])
                model = build_compile_and_fit_model(hyper_params, inner_train_set)
            else:
                # Continues fitting model from where previous 'fitting' left off.
                hyper_params["epochs"] = epochs_list[j] - epochs_list[j - 1]
                print("Number of epochs trained:", hyper_params["epochs"])
                model.fit(x=inner_train_set[0], y=inner_train_set[1], batch_size=hyper_params["batch-size"],
                          epochs=hyper_params["epochs"], verbose=0)

            # test on val set and record results (in order val sets are tested)
            # Done for each epochs value
            cur_train_results = compute_performance_measures(model, inner_train_set)
            cur_val_results = compute_performance_measures(model, inner_val_set)

            for performance_measure in all_val_results[0]:
                all_train_results[j][performance_measure].append(cur_train_results[performance_measure])
                all_val_results[j][performance_measure].append(cur_val_results[performance_measure])

            # Store copy of model
            models[j].append(copy(model))

    return all_train_results, all_val_results, models


def grid_search_cv(train_set, k, hyper_params_grid, fixed_length_for_time_series, data_augmentation):

    # Stores all hyper-parameter combinations to try out in grid search - EXCLUDING epochs
    hyper_param_combinations = []
    for hyper_param in hyper_params_grid:
        # Epochs are grid searched too, but in a different, more efficient way
        if hyper_param.lower() == "epochs":
            continue
        hyper_param_combinations.append(hyper_params_grid[hyper_param])
    hyper_param_combinations = list(itertools.product(*hyper_param_combinations))

    # For the two variables below, we have a list of dictionaries, one for each hyper-parameter combination (including epochs).
    # Each dictionary stores either train or val results, where the results are obtained in inner cv
    all_train_results = []
    all_val_results = []
    # Stores all hyper-param configurations, where epochs are included
    all_configurations = []
    # Stores all the trained models, for each configuration. Each configuration has (K_INNER) models.
    all_models = []

    for i in range(len(hyper_param_combinations)):

        # Extracting hyper-parameters to use, excluding epochs
        cur_hyper_params = list(hyper_param_combinations[i])
        hyper_params_dict = dict()
        index = 0
        for key in hyper_params_grid:
            if key.lower() == "epochs":
                continue
            hyper_params_dict[key] = cur_hyper_params[index]
            index += 1

        # cur_val_results returns the results for E fixed hyper-parameter configurations, where E is the # of
        # epochs values in the grid passed in (i.e a list of E dictionaries)
        cur_train_results, cur_val_results, cur_models = \
            regular_cv(train_set, k, hyper_params_dict, hyper_params_grid["epochs"], fixed_length_for_time_series, data_augmentation)

        # Store all results, for each of the E dictionaries (i.e configurations, where epochs is now fixed)
        for e in range(len(hyper_params_grid["epochs"])):
            all_train_results.append(cur_train_results[e])
            all_val_results.append(cur_val_results[e])
            # Adds info about the num of epochs to the given hyper-param combination. Added to front of list
            temp = cur_hyper_params.copy()
            temp.insert(0, hyper_params_grid["epochs"][e])
            all_configurations.append(temp)
            all_models.append(cur_models[e])

    # print("\nAll average validation results and corresponding, ordered configurations (best hyper-params used to do",
    #       "final training)")
    # print(all_average_val_results)
    # print(ordered_configurations)
    # exit()

    return all_train_results, all_val_results, all_configurations, all_models


def nested_cv(dataset, k_outer, k_inner, hyper_params_grid, fixed_length_for_time_series, data_augmentation=None):
    # Stores the results for all of the best hyper-parameters found
    all_best_val_results = performance_measures_dict()
    all_best_train_results = performance_measures_dict()
    all_test_results = performance_measures_dict()
    optimal_configurations = []

    hyper_params_grid["epochs"].sort()

    # OUTER CV - split data into k_outer folds. Each fold is a ParticipantsData object
    outer_folds = split_dataset_into_k_folds(dataset, k=k_outer)

    for i in range(k_outer):

        train_set = join_folds(outer_folds[0:i] + outer_folds[i + 1:])
        test_set = outer_folds[i]
        # print("Training set surgeries:", train_set.surgeries_stats["surgeries"])
        # print("Test set surgeries:", test_set.surgeries_stats["surgeries"], "\n")

        # GRID SEARCH
        train_results, val_results, all_configurations, all_models = \
            grid_search_cv(train_set, k_inner, hyper_params_grid, fixed_length_for_time_series, data_augmentation)

        # Get best model and corresponding results. No longer training a final, outer model
        model, hyper_params_index, model_index = get_best_model(val_results, all_configurations, all_models)
        best_hyper_params = all_configurations[hyper_params_index]
        train_results = train_results[hyper_params_index]
        val_results = val_results[hyper_params_index]

        print("\nBest hyper-params found:")
        print(best_hyper_params)

        # Prepare test set
        test_set = participants_storage_to_dictionary(test_set)
        change_format_of_time_series(test_set)
        zero_pad_time_series(test_set, fixed_length=fixed_length_for_time_series)
        test_set = ut.split_set_into_x_and_y(test_set)

        # Compute test set results - already have (inner) training results. Must recompile ?
        # model.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=best_hyper_params[5]),
        #     loss='binary_crossentropy',
        #     metrics=['binary_accuracy']
        # )
        cur_test_results = compute_performance_measures(model, test_set)

        print("\nCurrent train results:")
        for measure in train_results:
            print(measure + ":", train_results[measure][model_index])
        print("\nCurrent val results:")
        for measure in val_results:
            print(measure + ":", val_results[measure][model_index])
        print("\nCurrent test results:")
        for measure in cur_test_results:
            print(measure + ":", cur_test_results[measure])

        for performance_measure in all_test_results:
            all_best_train_results[performance_measure].append(train_results[performance_measure][model_index])
            all_best_val_results[performance_measure].append(val_results[performance_measure][model_index])
            all_test_results[performance_measure].append(cur_test_results[performance_measure])
        optimal_configurations.append(best_hyper_params)

        print("Done test " + str(i + 1) + "\\" + str(k_outer) + "...\n")

    return outer_folds, all_best_train_results, all_best_val_results, all_test_results, optimal_configurations


# Averaging related code:
# print("Current training results for some fixed hyper-param combination, for various epoch values:")
# print(cur_train_results)

# # The various lists of validation performance measures are averaged, for each of the E dictionaries
# cur_average_train_results = []
# cur_average_val_results = []
# for j in range(len(cur_val_results)):
#     cur_average_train_results.append(dict())
#     cur_average_val_results.append(dict())
#     for performance_measure in cur_val_results[j]:
#         cur_average_train_results[j][performance_measure] = mean(cur_train_results[j][performance_measure])
#         cur_average_val_results[j][performance_measure] = mean(cur_val_results[j][performance_measure])

# print("\nCurrent validation performances:")
# print(cur_val_results)

# print("\nAveraged results for a fixed hyper-param configuration: ")
# print(cur_average_val_results)
#
# print("\nall_average_val_results before appending:")
# print(all_average_val_results)

