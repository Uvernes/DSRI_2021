import itertools
import math
from statistics import mean
from sklearn.metrics import f1_score, roc_curve, auc, precision_score, recall_score, accuracy_score
import model_related.utils as ut
from model_related.model_methods import build_compile_and_fit_model
from utility.conversion_methods import *
from utility.k_folds_methods import split_dataset_into_k_folds,  join_folds


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

    # AUC - Sklearn
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    performance_results["AUC"] = auc(fpr, tpr)

    # get the best threshold - Youden’s J statistic method
    if find_optimal_threshold:
        J = tpr - fpr
        ix = np.argmax(J)
        threshold = thresholds[ix]
        # print('Best Threshold=%f' % threshold)

    y_pred_rounded = np.where(y_pred > threshold, 1, 0)

    # Binary crossentropy loss
    performance_results["Binary crossentropy loss"], prev_accuracy = \
        model.evaluate(x=x, y=y_true, verbose=0)

    # Binary accuracy
    performance_results["Binary accuracy"] = accuracy_score(y_true, y_pred_rounded)

    # print("Prev accuracy:", prev_accuracy)
    # print("New accuracy:", performance_results["Binary accuracy"])

    # Precision, Recall - Sklearn
    performance_results["Precision"] = precision_score(y_true, y_pred_rounded, zero_division=0)
    performance_results["Recall"] = recall_score(y_true, y_pred_rounded, zero_division=0)

    # f1-score
    performance_results["f1-score"] = f1_score(y_true, y_pred_rounded)

    # prev_y_pred_rounded = np.where(y_pred > 0.5, 1, 0)
    # print("Prev f1-score:", f1_score(y_true, prev_y_pred_rounded))
    # print("New f1-score:", performance_results["f1-score"])

    # AUC - tensorflow
    # m = tf.keras.metrics.AUC(num_thresholds=1000)
    # m.update_state(y_true, y_pred)
    # m.update_state([0, 0, 1, 1], [0, 0.5, 0.55, 0.6])
    # print(m.result().numpy())
    # print(float(m.result()))
    # performance_results["AUC"] = float(m.result())
    # print("tf AUC:", float(m.result()))

    # tensorflow - Precision
    # m = tf.keras.metrics.Precision()
    # m.update_state(y_true, y_pred_rounded)
    # performance_results["Precision"] = float(m.result().numpy())

    # tensorflow - Recall
    # m = tf.keras.metrics.Recall()
    # m.update_state(y_true, y_pred_rounded)
    # performance_results["Recall"] = float(m.result().numpy())

    if find_optimal_threshold:
        return performance_results, threshold
    return performance_results


# This function is rank based. No longer used
def get_best_hyper_params_OLD(all_val_results, ordered_configurations):
    # Compute the 'rank' for each configuration. 'rank' is the sum of the number of configurations that have better
    # results for a given performance measure,  squared, for each measure in selected_measure.
    # Thus, the lower the rank, the better

    # Measures we look at for determining rank
    selected_measures = ["Binary accuracy", "AUC", "f1-score"]
    ranks = []
    for i in range(len(ordered_configurations)):
        total_rank = 0
        for performance_measure in selected_measures:
            temp_rank = 0
            for j in range(len(all_val_results[performance_measure])):
                # When computing rank, do not compare configuration to itself
                if i == j:
                    continue
                # If some other configuration performs better for some measure, rank increases by 1
                if (all_val_results[performance_measure][i] < all_val_results[performance_measure][j]) and \
                        ("loss" not in performance_measure.lower()):
                    temp_rank += 1
                elif (all_val_results[performance_measure][i] > all_val_results[performance_measure][j]) and \
                        ("loss" in performance_measure.lower()):
                    temp_rank += 1
            # Squaring avoids 'imbalances' between temp_ranks
            total_rank += temp_rank ** 2
        ranks.append(total_rank)
    return ordered_configurations[ranks.index(min(ranks))]


# Score based system, but only considers validation results, not the inner train results
def get_best_hyper_params_OLD_2(all_val_results, all_configurations):
    """
    This function takes in all of the validation results from grid search and the corresponding configurations,
    and it returns what the best configuration of hyper-parameters is, based on these results
    :param all_val_results: dictionary storing all of the average results acquired from grid search, for each possible
                            hyper-param configuration. Specifically, each key stores a list of all the averages for some
                            performance measure, for all the different hyper-param configurations tested
    :param all_configurations: The i'th entry in any of the lists stored in val_results corresponds to the results
                                   given the i'th configuration in all_configurations
    :return: Best hyper-params (i.e one with highest score), considering the validation results
    """
    print("all_val_results:")
    for measure in all_val_results:
        print(measure + ":", all_val_results[measure])
    print("All configurations:")
    print(all_configurations)

    # Measures we look at for determining scores. Scores are the squared sum of these results
    selected_measures = ["Binary accuracy", "AUC", "f1-score"]
    scores = []
    for i in range(len(all_configurations)):
        cur_score = 0
        for performance_measure in selected_measures:
            # Square rooting gives preference to results that have less 'imbalances' between performance measures
            cur_score += math.sqrt(all_val_results[performance_measure][i])
        scores.append(cur_score)

    print("Scores:", scores, "\n")

    best_index = scores.index(max(scores))
    # Return index of hyper-parameters corresponding to the highest score (or first highest if there are ties)
    return all_configurations[best_index], best_index


def get_best_hyper_params_OLD_3(all_train_results, all_val_results, all_configurations):
    """
    This function takes in all of the validation results from grid search and the corresponding configurations,
    and it returns what the best configuration of hyper-parameters is, based on these results
    :param all_train_results: Analogous to all_val_results, but for the inner training set results
    :param all_val_results: dictionary storing all of the average results acquired from grid search, for each possible
                            hyper-param configuration. Specifically, each key stores a list of all the averages for some
                            performance measure, for all the different hyper-param configurations tested
    :param all_configurations: The i'th entry in any of the lists stored in val_results corresponds to the results
                                   given the i'th configuration in all_configurations
    :return: Best hyper-params (i.e one with highest score), considering the validation results
    """
    print("\nall_train_results (averages):")
    for measure in all_train_results:
        print(measure + ":", all_train_results[measure])
    print("\nall_val_results (averages):")
    for measure in all_val_results:
        print(measure + ":", all_val_results[measure])
    print("\nAll configurations:")
    print(all_configurations)

    # Measures we look at for determining scores. Scores are the squared rooted, sum of these results
    selected_measures = ["Binary accuracy", "AUC", "f1-score"]
    scores = []
    for i in range(len(all_configurations)):
        cur_score = 0
        for performance_measure in selected_measures:
            # Square rooting gives preference to results that have less 'imbalances' between performance measures
            cur_train_score = math.sqrt(all_train_results[performance_measure][i])
            cur_val_score = math.sqrt(all_val_results[performance_measure][i])
            cur_score += cur_train_score + cur_val_score
        # Scores are normalized so they fall in the range [0,1] - doesn't affect which hyper-params chosen since
        # scores all multiplied by the same scalar
        scores.append((cur_score / len(selected_measures)) / 2)

    print("\nScores:\n", scores, "\n")

    best_index = scores.index(max(scores))
    # Return index of hyper-parameters corresponding to the highest score (or first highest if there are ties)
    return all_configurations[best_index], best_index


def get_best_hyper_params(all_val_results, all_configurations):
    """
    This function takes in all of the validation results from grid search and the corresponding configurations,
    and it returns what the best configuration of hyper-parameters is, based on these results
    :param all_val_results: dictionary storing all of the average results acquired from grid search, for each possible
                            hyper-param configuration. Specifically, each key stores a list of all the averages for some
                            performance measure, for all the different hyper-param configurations tested
    :param all_configurations: The i'th entry in any of the lists stored in val_results corresponds to the results
                                   given the i'th configuration in all_configurations
    :return: Best hyper-params (i.e one with highest score), considering the validation results
    """
    print("\nall_val_results (averages):")
    for measure in all_val_results:
        print(measure + ":", all_val_results[measure])
    print("\nAll configurations:")
    print(all_configurations)

    # Measures we look at for determining scores. Scores are the squared rooted, sum of these results
    selected_measures = ["Binary accuracy", "AUC", "f1-score"]
    scores = []
    for i in range(len(all_configurations)):
        cur_score = 0
        for performance_measure in selected_measures:
            # Square rooting gives preference to results that have less 'imbalances' between performance measures
            cur_score += math.sqrt(all_val_results[performance_measure][i])
            # Scores are normalized so they fall in the range [0,1] - doesn't affect which hyper-params chosen since
            # scores all multiplied by the same scalar
        scores.append((cur_score / len(selected_measures)))

    print("\nScores:\n", scores, "\n")

    best_index = scores.index(max(scores))
    # Return index of hyper-parameters corresponding to the highest score (or first highest if there are ties)
    return all_configurations[best_index], best_index


# i.e non-nested cv. Computes all performance measures of interest, for each fold, for various # of epochs
def regular_cv(train_set, k, hyper_params, epochs_list, fixed_length_for_time_series, data_augmentation):
    # We have E validation results dictionaries, where E = len(epochs_list). Same for training results
    all_train_results = []
    all_val_results = []
    optimal_thresholds = []
    for _ in epochs_list:
        all_val_results.append(performance_measures_dict())
        all_train_results.append(performance_measures_dict())
        optimal_thresholds.append(0.0)

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
            cur_val_results, optimal_threshold = \
                compute_performance_measures(model, inner_val_set, find_optimal_threshold=True)
            cur_train_results = compute_performance_measures(model, inner_train_set, threshold=optimal_threshold)
            # print("\nResults for inner training set, for different number of epochs:\n", cur_train_results)
            # print("\nResults for val set, for different number of epochs:\n", cur_val_results)

            for performance_measure in all_val_results[0]:
                all_train_results[j][performance_measure].append(cur_train_results[performance_measure])
                all_val_results[j][performance_measure].append(cur_val_results[performance_measure])

            # Optimal threshold for some configuration of hyper-params is the average of the ones found using each of
            # the k val sets
            optimal_thresholds[j] += optimal_threshold / k

    return all_train_results, all_val_results, optimal_thresholds


def grid_search_cv(train_set, k, hyper_params_grid, fixed_length_for_time_series, data_augmentation):
    # We do some sorting based on measure below. Doesn't effect anything, just for visual reasons
    val_performance_measure = "Binary accuracy"

    # Stores all hyper-parameter combinations to try out in grid search - EXCLUDING epochs
    hyper_param_combinations = []
    for hyper_param in hyper_params_grid:
        # Epochs are grid searched too, but in a different, more efficient way
        if hyper_param.lower() == "epochs":
            continue
        hyper_param_combinations.append(hyper_params_grid[hyper_param])
    hyper_param_combinations = list(itertools.product(*hyper_param_combinations))
    # print("Combinations:")
    # print(hyper_param_combinations)
    # exit()

    all_average_val_results = performance_measures_dict()
    all_average_train_results = performance_measures_dict()
    ordered_configurations = []
    all_optimal_thresholds = []

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
        cur_train_results, cur_val_results, cur_optimal_thresholds = \
            regular_cv(train_set, k, hyper_params_dict, hyper_params_grid["epochs"], fixed_length_for_time_series, data_augmentation)

        # print("Current training results for some fixed hyper-param combination, for various epoch values:")
        # print(cur_train_results)

        # The various lists of validation performance measures are averaged, for each of the E dictionaries
        cur_average_train_results = []
        cur_average_val_results = []
        for j in range(len(cur_val_results)):
            cur_average_train_results.append(dict())
            cur_average_val_results.append(dict())
            for performance_measure in cur_val_results[j]:
                cur_average_train_results[j][performance_measure] = mean(cur_train_results[j][performance_measure])
                cur_average_val_results[j][performance_measure] = mean(cur_val_results[j][performance_measure])

        # print("\nCurrent validation performances:")
        # print(cur_val_results)

        # print("\nAveraged results for a fixed hyper-param configuration: ")
        # print(cur_average_val_results)
        #
        # print("\nall_average_val_results before appending:")
        # print(all_average_val_results)

        # Store all results, for each of the E dictionaries
        for e in range(len(hyper_params_grid["epochs"])):
            # Stores the average results for some fixed hyper-parameter combination, where # of epochs is also fixed
            single_train_results_dictionary = cur_average_train_results[e]
            single_val_results_dictionary = cur_average_val_results[e]
            # Add current results to all results in descending order, where we descend w.r.t val_performance_measure
            j = 0
            # e.g if val_performance_measure is "AUC", variable below stores a list of AUC averages
            measures_of_interest = all_average_val_results[val_performance_measure]
            while j < len(measures_of_interest):
                if (single_val_results_dictionary[val_performance_measure] > measures_of_interest[j]) and \
                        ("loss" not in val_performance_measure):
                    break
                # Accounts for fact that unlike other performance measures, we want loss to be minimized, not maximized
                if (single_val_results_dictionary[val_performance_measure] < measures_of_interest[j]) and \
                        ("loss" in val_performance_measure):
                    break
                j += 1
            for performance_measure in all_average_val_results:
                all_average_train_results[performance_measure].insert(j, single_train_results_dictionary[
                    performance_measure])
                all_average_val_results[performance_measure].insert(j,
                                                                    single_val_results_dictionary[performance_measure])
            # Adds info about the num of epochs to the given hyper-param combination. Added to front of list
            temp = cur_hyper_params.copy()
            temp.insert(0, hyper_params_grid["epochs"][e])
            ordered_configurations.insert(j, temp)
            all_optimal_thresholds.insert(j, cur_optimal_thresholds[e])

    # print("\nAll average validation results and corresponding, ordered configurations (best hyper-params used to do",
    #       "final training)")
    # print(all_average_val_results)
    # print(ordered_configurations)
    # exit()

    return all_average_train_results, all_average_val_results, ordered_configurations, all_optimal_thresholds


def nested_cv(dataset, k_outer, k_inner, hyper_params_grid, fixed_length_for_time_series, data_augmentation=None):
    # Stores the average validation results for all of the best hyper-parameters found
    all_best_val_results = performance_measures_dict()
    all_best_inner_train_results = performance_measures_dict()
    all_train_results = performance_measures_dict()
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
        inner_train_results, inner_val_results, ordered_configurations, optimal_thresholds = \
            grid_search_cv(train_set, k_inner, hyper_params_grid, fixed_length_for_time_series, data_augmentation)

        # print("\nFinished grid search...")
        # print("\nValidation results:")
        # for performance_measure in inner_val_results:
        #     print(performance_measure + ":", inner_val_results[performance_measure])
        # print("\nCorresponding Hyper-parameter configurations:")
        # print(ordered_configurations)

        best_hyper_params, best_index = get_best_hyper_params(inner_val_results, ordered_configurations)
        optimal_threshold = optimal_thresholds[best_index]

        print("\nBest hyper-params found:")
        print(best_hyper_params)
        print("Optimal threshold:", optimal_threshold, "\n")
        # exit()

        # Prepare train and test sets
        train_set = participants_storage_to_dictionary(train_set)
        test_set = participants_storage_to_dictionary(test_set)

        # ------- Apply data augmentation to training set here ------ #
        if data_augmentation is not None:
            # train_set enhanced
            synthetic_dataset = data_augmentation.execute(train_set)
            train_set = join_dataset_dictionaries(train_set, synthetic_dataset)

        change_format_of_time_series(train_set)
        change_format_of_time_series(test_set)

        zero_pad_time_series(train_set, fixed_length=fixed_length_for_time_series)
        zero_pad_time_series(test_set, fixed_length=fixed_length_for_time_series)

        train_set = ut.split_set_into_x_and_y(train_set)
        test_set = ut.split_set_into_x_and_y(test_set)

        # Fit model on entire training set
        # best_hyper_params = ordered_configurations[0]
        hyper_params_dict = dict()
        index = 0
        for key in hyper_params_grid:
            hyper_params_dict[key] = best_hyper_params[index]
            index += 1

        print("Hyper params used to train outer model:", hyper_params_dict)
        model = build_compile_and_fit_model(hyper_params_dict, train_set)

        # Compute training set and test set results
        # print("\n\nMaking predictions on test set using model with optimal hyper-parameters...")
        cur_train_results = compute_performance_measures(model, train_set, threshold=optimal_threshold)
        cur_test_results = compute_performance_measures(model, test_set, threshold=optimal_threshold)
        print("\nResults for outer training set:\n", cur_train_results)

        # print("\nRetraining model, expecting ~same training results...")
        # model = build_compile_and_fit_model(hyper_params_dict, train_set)
        # cur_train_results = compute_performance_measures(model, train_set)
        # print("\nRepeated results for outer training set:\n", cur_train_results)

        print("\nResults for test set:\n", cur_test_results)

        # print("Average validation results for best hyper-params found:")
        # for measure in inner_val_results:
        #     print(measure + ":", inner_val_results[measure][best_index])
        #
        # print("\nCurrent train results:")
        # for measure in cur_train_results:
        #     print(measure + ":", cur_train_results[measure])
        #
        # print("\nCurrent test results:")
        # for measure in cur_test_results:
        #     print(measure + ":", cur_test_results[measure])
        # print()

        for performance_measure in all_test_results:
            all_best_inner_train_results[performance_measure].append(inner_train_results[performance_measure][best_index])
            all_best_val_results[performance_measure].append(inner_val_results[performance_measure][best_index])
            all_train_results[performance_measure].append(cur_train_results[performance_measure])
            all_test_results[performance_measure].append(cur_test_results[performance_measure])
        optimal_configurations.append(best_hyper_params)

        print("Done test " + str(i + 1) + "\\" + str(k_outer) + "...\n")

    return outer_folds, all_best_inner_train_results, all_best_val_results, all_train_results, all_test_results, \
        optimal_configurations
