import math
import random
import itertools
import model_related.classes
import model_related.utils as ut
import numpy as np
from utility.conversion_methods import *
from model_related.model_methods import build_compile_and_fit_model, auc_internal
from sklearn.metrics import recall_score, confusion_matrix, f1_score
from statistics import mean

# Note: If we get poor splitting, logic in here will need to change

def compare_ratios(ratios_1, ratios_2, novices_to_experts, novices_IP_to_OOP, experts_IP_to_OOP):
    """
    This function takes in two lists of ratios, and returns True if the first list of ratios is closer to the desired
    ratios than the second. False returned otherwise
    List of ratios are of the following form:
    [[fold_1 novices:experts, fold_1 novices_IP:novices_OOP, fold_1 experts_IP:experts_OOP],
     [fold_2 novices:experts, fold_2 novices_IP:novices_OOP, fold_2 experts_IP:experts_OOP]]
    Note: If two lists have the same IP to OOP ratios, then the one with the better novices:experts ratio is returned.
    """
    ratios = [ratios_1, ratios_2]

    # Measures how close ratios_1 and ratios_2 IP:OOP ratios are to the desired ratios, where both folds in ratios_1,
    # ratios_2, are considered.
    # Note: Terms are squared to ensure both ratios at a given level try to get close to the desired ratio, not just
    # one of them
    ratios_IP_to_OOP_dif = []
    for i in range(2):

        ratios_IP_to_OOP_dif.append(
            (((ratios[i][0][1] - novices_IP_to_OOP) / novices_IP_to_OOP)**2 +
             ((ratios[i][0][2] - experts_IP_to_OOP) / experts_IP_to_OOP)**2)**2
            +
            (((ratios[i][1][1] - novices_IP_to_OOP) / novices_IP_to_OOP)**2 +
             ((ratios[i][1][2] - experts_IP_to_OOP) / experts_IP_to_OOP)**2)**2)

    if ratios_IP_to_OOP_dif[0] < ratios_IP_to_OOP_dif[1]:
        return True
    elif ratios_IP_to_OOP_dif[0] > ratios_IP_to_OOP_dif[1]:
        return False

    # Case where scores are equal
    # Check which list of ratios improves novices:experts most
    r1_novices_to_experts_dif = (ratios_1[0][0] - novices_to_experts)**2 + (ratios_1[1][0] - novices_to_experts)**2
    r2_novices_to_experts_dif = (ratios_2[0][0] - novices_to_experts)**2 + (ratios_2[1][0] - novices_to_experts)**2
    if r1_novices_to_experts_dif < r2_novices_to_experts_dif:
        return True
    else:
        return False


def new_ratios_if_participants_swapped(fold_1, fold_2, p1_name, p2_name):
    """
    This function computes what the new ratios in each fold would be if p1 and p2 were swapped
    :param fold_1: First fold being considered
    :param fold_2: Second fold being considered
    :param p1_name: Participant in fold_1 to swap
    :param p2_name: Participant in fold_2 to swap
    :return: A list storing what the new ratios in each fold would be if p1 and p2 were swapped. Of the form:
             [[fold_1 ratio 1, ... , fold_1 ratio 3] [fold_2 ratio 1, ... , fold_2 ratio 3 ]]

    """
    folds = [fold_1, fold_2]
    names = [p1_name, p2_name]
    new_ratios = [[], []]
    skill_levels = ["novices", "experts"]
    surgery_types = ["IP", "OOP"]
    # Each entry below stores stats for one of the 2 folds. It stores, in order:
    # new_novices_IP, new_novices_OOP, new_experts_IP, new_experts_OOP
    stats = [[], []]

    # print("\nNumber of novices_IP in fold 1, fold 2, before switch:")
    # print(fold_1.surgeries_stats["novices_IP"], "|", fold_2.surgeries_stats["novices_IP"])

    # Finds new stats for each fold
    for i in range(2):
        for j in range(len(skill_levels)):
            for k in range(len(surgery_types)):
                key = skill_levels[j] + "_" + surgery_types[k]
                stats[i].append(folds[i].surgeries_stats[key]
                                - folds[i].participants[names[i]].surgeries_stats[key]
                                + folds[(i+1) % 2].participants[names[(i+1) % 2]].surgeries_stats[key])

    # print("Number of novices_IP in fold 1, fold 2, after switch:")
    # print(stats[0][0], "|", stats[1][0])

    # Compute new ratios
    for i in range(2):

        # novices:experts
        num_novices = stats[i][0] + stats[i][1]
        num_experts = stats[i][2] + stats[i][3]
        if num_experts == 0:
            new_ratios[i].append(math.inf)
        else:
            new_ratios[i].append(num_novices / num_experts)
        # novices_IP:novices:OOP
        if stats[i][1] == 0:
            new_ratios[i].append(math.inf)
        else:
            new_ratios[i].append(stats[i][0] / stats[i][1])
        # experts_IP:experts:OOP
        if stats[i][3] == 0:
            new_ratios[i].append(math.inf)
        else:
            new_ratios[i].append(stats[i][2] / stats[i][3])

        # print("Fold #" + str(i) + "'s new ratios:", new_ratios[i])

    return new_ratios


def find_best_participants_to_swap_between_folds(fold_1, fold_2, novices_to_experts, novices_IP_to_OOP,
                                                 experts_IP_to_OOP):
    """
    Helper function. This returns the pair of participants to swap between two given folds in order to best improve
    similiarity of the ratios in the fold to the given ratios passed in.
    :param fold_1: fold 1 of 2 we are considering
    :param fold_2: fold 2 of 2 we are considering
    :param novices_to_experts: novices to experts ratio in original dataset
    :param novices_IP_to_OOP:  novices IP to OOP ratio in original dataset
    :param experts_IP_to_OOP:  experts IP to OOP ratio in original dataset
    :return: Info regarding the pair of participants found. It is of the following form:
        [p1 name (in fold 1) , p2 name (in fold 2)], [[fold_1 ratio_1, ..., fold_1 ratio_3], [fold_2 ratio_1, ...]]
        NOTE: if no pair found, returns []
    """
    fold_1_ratios = [fold_1.novices_to_experts_ratio(), fold_1.novices_IP_to_OOP_ratio(),
                     fold_1.experts_IP_to_OOP_ratio()]
    fold_2_ratios = [fold_2.novices_to_experts_ratio(), fold_2.novices_IP_to_OOP_ratio(),
                     fold_2.experts_IP_to_OOP_ratio()]
    best_pair = []
    best_ratios = []
    for p1_name in fold_1.participants:
        for p2_name in fold_2.participants:
            cur_ratios = new_ratios_if_participants_swapped(fold_1, fold_2, p1_name, p2_name)
            # The novices:experts ratio must improve in both folds, o.w cur_ratios is not considered
            if (abs(novices_to_experts - cur_ratios[0][0]) >= abs(novices_to_experts - fold_1_ratios[0])) or \
                    (abs(novices_to_experts - cur_ratios[1][0]) >= abs(novices_to_experts - fold_2_ratios[0])):
                # print("Worse than original")
                continue

            # Compare cur_ratios to best_ratios . If it is better (or best_pair is empty), update
            # best_pairs, best_ratios.
            better = True if (not best_pair) else compare_ratios(cur_ratios, best_ratios, novices_to_experts,
                                                                 novices_IP_to_OOP, experts_IP_to_OOP)
            if better:
                best_ratios = cur_ratios
                best_pair = [p1_name, p2_name]
                # print("Better than cur best")

    return best_pair, best_ratios


def swap_participants_between_all_folds(folds, novices_to_experts, novices_IP_to_OOP, experts_IP_to_OOP):
    """
    Helper function. Called after dataset is split into k user-out folds. It swaps participants between folds such
    that the expert to novice ratio is closer to that of the original dataset, and potentially that of the
    novice_IPP:novice_OPP, expert_IP:expert_OPP ratios.
    :param folds: k user-out folds
    :param novices_to_experts: novices to experts ratio in original dataset
    :param novices_IP_to_OOP:  novices IP to OOP ratio in original dataset
    :param experts_IP_to_OOP:  experts IP to OOP ratio in original dataset
    :return: Number of swaps done
    """

    num_swaps = 0
    for i in range(len(folds)):

        # This stores the two participants to swap. It is of the following form:
        # [[fold p1 is in, p1 name], [fold p2 is in, p2 name]]
        best_pair = []

        # This stores what the new ratios in each fold will be after the swap using best_pair. Of the form:
        # [[fold i ratio 1, ... , fold i ratio 3] [fold j ratio 1, ... , fold j ratio 3 ]]
        best_ratios = []

        for j in range(i+1, len(folds)):
            # cur_pair is a possible swap between fold i and some other fold that improves the ratio(s).
            # If it improves it more than the current best_pair, it replaces it
            cur_pair, cur_ratios = find_best_participants_to_swap_between_folds(folds[i], folds[j], novices_to_experts,
                                                                                novices_IP_to_OOP, experts_IP_to_OOP)
            # Checks if pair not found
            if not cur_pair:
                continue

            # Compare cur_ratios to best_ratios . If it is better, update best_pairs, best_ratios
            better = True if (not best_pair) else compare_ratios(cur_ratios, best_ratios, novices_to_experts,
                                                                 novices_IP_to_OOP, experts_IP_to_OOP)
            if better:
                best_ratios = cur_ratios
                best_pair = [[i, cur_pair[0]], [j, cur_pair[1]]]

        # Checks if best_pair was found
        if best_pair:
            # perform swap
            fold_1 = folds[best_pair[0][0]]
            fold_2 = folds[best_pair[1][0]]
            # print("Fold 1 ratios before swap:")
            # print(fold_1.novices_to_experts_ratio(), "|", fold_1.novices_IP_to_OOP_ratio(),
            #       "|", fold_1.experts_IP_to_OOP_ratio())
            p1 = fold_1.participants[best_pair[0][1]]
            # p2 = fold_2.participants[list(fold_2.participants.keys())[random.randint(0, len(fold_2.participants) - 1)]]
            p2 = fold_2.participants[best_pair[1][1]]
            # print("p1, p2 names, respectively:", p1.name, "|", p2.name)
            fold_2.add_participant(fold_1.pop_participant(p1.name))
            fold_1.add_participant(fold_2.pop_participant(p2.name))
            num_swaps += 1
            # print("Fold 1 ratios after swap:")
            # print(fold_1.novices_to_experts_ratio(), "|", fold_1.novices_IP_to_OOP_ratio(),
            #       "|", fold_1.experts_IP_to_OOP_ratio())

    return num_swaps


def split_dataset_into_k_folds(dataset, k, shuffle=True, recursive=True):

    """
    This function takes in a dataset and splits it into k user-out, stratified folds.
    :param dataset: Dataset to split into k folds. Represented as a ParticipantsStorage object
    :param k: Number of folds
    :param shuffle: If true, randomly shuffles the dataset before splitting it into k folds
    :param recursive: If True, then if the dataset is split into k folds in an unsatisfactory manner, than this
                    function is called again, recursively
    :return: list of length k, where each element is a fold represented by a ParticipantsStorage object
    """

    k = int(k)
    if k < 2:
        return dataset

    # Split dataset into k user-out folds. Done randomly if shuffle set to True
    folds = []
    participant_names = list(dataset.participants.keys())
    if shuffle:
        random.shuffle(participant_names)
    for i in range(k):
        folds.append(model_related.classes.ParticipantsStorage())
        # Add ~n/k participants to the i'th fold. Last fold left with remainder
        upper_index = (i + 1) * math.floor(len(participant_names)/k) if i < k-1 else len(participant_names)
        for j in range(i * math.floor(len(participant_names)/k), upper_index):
            folds[i].add_participant(dataset.participants[participant_names[j]])

    # Spread out remainder from last fold, starting at first fold
    remainder = len(participant_names) % k
    for i in range(remainder):
        folds[i].add_participant(folds[k-1].pop_participant())

    # print("Number of participants in each fold:")
    # for i in range(k):
    #     print("Fold #" + str(i+1) + ":", len(folds[i].participants))

    # At this point we have user-out folds. Below we deal with stratification of Novice:Expert,
    # novices_IP:novices_OOP, and experts_IP:experts_OOP, with Novice:Expert being prioritized.

    total_num_swaps = 0
    while True:
        cur_num_swaps = \
            swap_participants_between_all_folds(folds, dataset.novices_to_experts_ratio(),
                                                dataset.novices_IP_to_OOP_ratio(), dataset.experts_IP_to_OOP_ratio())
        if cur_num_swaps == 0:
            break
        total_num_swaps += cur_num_swaps

    # We check to see that each fold has at least 1 of each skill_level, surgery_type pair (i,e are 'good_folds').
    # If not, we recursively call this method again
    # Note: Could also have this occur if folds are not balanced well in general (optional)
    good_folds = True
    for fold in folds:
        for skill_level in ["novices", "experts"]:
            for surgery_type in ["IP", "OOP"]:
                if fold.surgeries_stats[skill_level + "_" + surgery_type] == 0:
                    good_folds = False

    if not good_folds and recursive:
        folds = split_dataset_into_k_folds(dataset, k, shuffle=True)

    # --- Debugging / testing related code: ---- #

    # print("\nNumber of swaps done:", total_num_swaps)
    #
    # print("\nnovices:experts, novices_IP:novices_OOP, experts_IP:experts_OOP in original dataset, respectively:")
    # print(dataset.novices_to_experts_ratio(), "|", dataset.novices_IP_to_OOP_ratio(), "|",
    #       dataset.experts_IP_to_OOP_ratio())
    #
    # print("\nnovices:experts, novices_IP:novices_OOP, experts_IP:experts_OOP in respectively, for each fold:")
    # for i in range(k):
    #     print("Fold #" + str(i+1) + ":", folds[i].novices_to_experts_ratio(), "|", folds[i].novices_IP_to_OOP_ratio(),
    #           "|", folds[i].experts_IP_to_OOP_ratio())
    #
    # print("\nNumber of novices IP, OOP, experts IP, OOP, respectively, for each fold")
    # for i in range(k):
    #     print("Fold #" + str(i+1) + ":", folds[i].surgeries_stats["novices_IP"], "|",
    #           folds[i].surgeries_stats["novices_OOP"], "|", folds[i].surgeries_stats["experts_IP"], "|",
    #           folds[i].surgeries_stats["experts_OOP"])
    # ------------------------------------ #

    return folds


def join_folds(folds):
    """
    :param folds - list of folds (i.e ParticipantsStorage objects)
    :return: A single ParticipantsStorage object corresponding to all the given folds joined together
    """
    result = model_related.classes.ParticipantsStorage()
    for fold in folds:
        for participant in fold.participants.values():
            result.add_participant(participant)
    return result


# i.e non-nested cv
def regular_cv(train_set, k, hyper_params, fixed_length_for_time_series):

    val_results = []
    inner_folds = split_dataset_into_k_folds(train_set, k)
    for i in range(k):

        inner_train_set = join_folds(inner_folds[0:i] + inner_folds[i+1:])
        inner_val_set = inner_folds[i]

        print("Inner training set surgeries:", inner_train_set.surgeries_stats["surgeries"])
        print("Inner validation set surgeries:", inner_val_set.surgeries_stats["surgeries"])

        # Convert sets (ParticipantsStorage objects) into dictionaries of time series
        inner_train_set = participants_storage_to_dictionary(inner_train_set)
        inner_val_set = participants_storage_to_dictionary(inner_val_set)

        print("Number of novices IP, OOP, experts IP, OOP, respectively in inner training set:")
        print(len(inner_train_set["Novices"]["OOP"]), "|", len(inner_train_set["Novices"]["IP"]), "|",
              len(inner_train_set["Experts"]["IP"]), "|", len(inner_train_set["Experts"]["OOP"]))

        # Apply data augmentation here to training set - LATER
        # potentially on val set too
        # e.g augment_training_set(inner_train_set)

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

        model = build_compile_and_fit_model(hyper_params, inner_train_set)

        print("\n")

        # test on val set and record results (in descending order)
        y_pred = model.predict(inner_val_set[0])

        # Testing AUC
        # auc = auc_internal(y_true=inner_val_set[1], y_pred=y_pred)
        # print("AUC: ", auc)
        # exit()

        y_pred = np.argmax(y_pred, axis=1)
        cur_val_result = f1_score(y_true=inner_val_set[1], y_pred=y_pred)

        if len(val_results) == 0:
            val_results.append(cur_val_result)
            continue
        for j in range(len(val_results)):
            if cur_val_result > val_results[j]:
                val_results.insert(j, cur_val_result)
                break
            if j == (len(val_results) - 1):
                val_results.append(cur_val_result)

    return val_results


def grid_search_cv(train_set, k, hyper_params_grid, fixed_length_for_time_series):

    # Stores all hyper-parameter combinations to try out in grid search
    hyper_param_combinations = []
    for hyper_param in hyper_params_grid.values():
        hyper_param_combinations.append(hyper_param)
    hyper_param_combinations = list(itertools.product(*hyper_param_combinations))

    average_val_results = []
    ordered_configurations = []
    for i in range(len(hyper_param_combinations)):

        # Extracting hyper-parameters to use
        cur_hyper_params = hyper_param_combinations[i]
        hyper_params_dict = dict()
        index = 0
        for key in hyper_params_grid:
            hyper_params_dict[key] = cur_hyper_params[index]
            index += 1

        # Only interested in the average val f1-score, AUC, etc.
        cur_average_val_result = regular_cv(train_set, k, hyper_params_dict, fixed_length_for_time_series)
        print("\nf1 results for a fixed hyper-param configuration: ")
        print(cur_average_val_result)

        cur_average_val_result = mean(cur_average_val_result)

        if len(average_val_results) == 0:
            average_val_results.append(cur_average_val_result)
            ordered_configurations.append(cur_hyper_params)
        else:
            # Adds in validation results such that they are in descending order
            for j in range(len(average_val_results)):
                if cur_average_val_result > average_val_results[j]:
                    average_val_results.insert(j, cur_average_val_result)
                    ordered_configurations.insert(j, cur_hyper_params)
                    break
                if j == (len(average_val_results) - 1):
                    average_val_results.append(cur_average_val_result)
                    ordered_configurations.append(cur_hyper_params)

    return average_val_results, ordered_configurations


def nested_cv(dataset, k_outer, k_inner, hyper_params_grid, fixed_length_for_time_series):

    test_results = []
    optimal_configurations = []

    # OUTER CV - split data into k_outer folds. Each fold is a ParticipantsData object
    outer_folds = split_dataset_into_k_folds(dataset, k=k_outer)

    for i in range(k_outer):

        train_set = join_folds(outer_folds[0:i] + outer_folds[i+1:])
        test_set = outer_folds[i]
        print("Training set surgeries:", train_set.surgeries_stats["surgeries"])
        print("Test set surgeries:", test_set.surgeries_stats["surgeries"], "\n")

        # GRID SEARCH
        inner_val_results, ordered_configurations =\
            grid_search_cv(train_set, k_inner, hyper_params_grid, fixed_length_for_time_series)

        print("\nFinished grid search...")
        print("Validation results:")
        print(inner_val_results)
        print("Corresponding Hyper-parameter configurations:")
        print(ordered_configurations)
        print("\n")

        # Prepare train and test sets
        train_set = participants_storage_to_dictionary(train_set)
        test_set = participants_storage_to_dictionary(test_set)

        change_format_of_time_series(train_set)
        change_format_of_time_series(test_set)

        zero_pad_time_series(train_set, fixed_length=fixed_length_for_time_series)
        zero_pad_time_series(test_set, fixed_length=fixed_length_for_time_series)

        train_set = ut.split_set_into_x_and_y(train_set)
        test_set = ut.split_set_into_x_and_y(test_set)
        x_test, y_test = test_set

        # Fit model on entire training set and test on test set
        best_hyper_params = ordered_configurations[0]
        hyper_params_dict = dict()
        index = 0
        for key in hyper_params_grid:
            hyper_params_dict[key] = best_hyper_params[index]
            index += 1
        model = build_compile_and_fit_model(hyper_params_dict, train_set)
        print("\n")

        print("\nMaking predictions on test set using model with optimal hyper-parameters...")
        y_pred = model.predict(x_test)
        # print(y_pred)
        y_pred = np.argmax(y_pred, axis=1)  # changes predictions to be single outputs, 0 or 1

        # Issue: all predictions are experts. Likely due to data becoming too skewed after window slicing

        # ---------- DEBUGGING ----------- #
        # print("\nAfter making predictions...")
        # print("\nTest set size:", len(y_test))
        # print("\nPredictions size:", len(y_pred))
        # num_pred_novices = 0
        # num_pred_experts = 0
        # for pred in y_pred:
        #     if pred == 1:
        #         num_pred_experts += 1
        #     if pred == 0:
        #         num_pred_novices += 1
        # print("\nNumber of predicted novices:", num_pred_novices)
        # print("Number of predicted experts:", num_pred_experts)
        #
        # num_test_novices = 0
        # num_test_experts = 0
        # for pred in y_test:
        #     if pred == 1:
        #         num_test_experts += 1
        #     if pred == 0:
        #         num_test_novices += 1
        # print("\nNumber of novices in test set:", num_test_novices)
        # print("Number of experts in test set:", num_test_experts)
        # ---------------------------------#

        cur_test_f_score = f1_score(y_true=y_test, y_pred=y_pred)

        # print("\nF1 score on test set:")
        # print("F1 score:", test_f_score)

        # Adds in test results such that they are in descending order
        if len(test_results) == 0:
            test_results.append(cur_test_f_score)
            optimal_configurations.append(best_hyper_params)
            continue
        for j in range(len(test_results)):
            if cur_test_f_score > test_results[j]:
                test_results.insert(j, cur_test_f_score)
                optimal_configurations.insert(j, best_hyper_params)
                break
            if j == (len(test_results) - 1):
                test_results.append(cur_test_f_score)
                optimal_configurations.append(best_hyper_params)

    return test_results, optimal_configurations


