import tslearn
import math
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sn
import model_related.utils as ut
from data_augmentation.jittering import apply_jittering_to_dataset
from data_augmentation.smote_based_wDBA import smote_based_weighted_dba
from model_related.classes import ProficiencyLabel
from model_related.model_methods import build_compile_and_fit_model
from utility.data_extraction_and_storage_methods import load_dataset
from utility.conversion_methods import *

from sklearn.metrics import recall_score, confusion_matrix, f1_score
from matplotlib import pyplot as plt

DATASET_PATH = r"C:\Users\uvern\Dropbox\My PC (LAPTOP-554U8A6N)\Documents\DSRI\Data\usneedle_data\usneedle_data"
SEQUENCE_TYPE = "NeedleTipToReference"
SLICE_WINDOW = 70

HYPER_PARAMETERS = {
    "kernel-size":      5,
    "filters":          64,
    "epochs":           1,    # originally 300
    "batch-size":       32,
    "dropout-rate":     0.5,
    "learning-rate":    0.0001,
    "regularizer":      tf.keras.regularizers.l1_l2(0.05)
}


# This is where we create our model, and train it using a dataset enhanced via data augmentation

# NOTE - labels: Novice = 0, Expert = 1

def main():

    # Load use_needle dataset. Stored as a list with 2 elements, where the first element stores all novice data,
    # second stores all the expert data
    dataset, _ = load_dataset(DATASET_PATH, SEQUENCE_TYPE)

    # ------ Testing where we reduce novice data size ------- #
    # while len(dataset[0]) > len(dataset[1]):
    #     dataset[0].pop()

    # 240 novices, 15 experts
    print("Number of novice and expert time series, respectively:", len(dataset[0]), len(dataset[1]))

    # Split dataset ~equally into training, validation, and test set. Each set gets an approximately equal prop.
    # of novices and experts
    random.shuffle(dataset[0])
    random.shuffle(dataset[1])

    novices_per_set = math.floor(len(dataset[0]) / 3)
    experts_per_set = math.floor(len(dataset[1]) / 3)
    train_set = [dataset[0][0:novices_per_set], dataset[1][0:experts_per_set]]
    val_set = [dataset[0][novices_per_set: 2*novices_per_set], dataset[1][experts_per_set:2*experts_per_set]]
    test_set = [dataset[0][2*novices_per_set:], dataset[1][2*experts_per_set:]]


    # --- DATA AUGMENTATION GOES HERE --- #
    # Note: Currently applying data augmentation to minority class only (in training set).
    # Results in going from 5 to ~80 (75) experts (hardcoded for now)

    # print("Number of training set experts before data augmentation:", len(train_set[1]))
    #
    # # Apply SMOTE-based wDBA
    # synthetic_experts = smote_based_weighted_dba(train_set[1], N=700, k=3)
    #
    # # Fix all rotation matrices
    # for i in range(len(synthetic_experts)):
    #     fix_rotation_matrices(synthetic_experts[i])
    #
    # # Create additional synthetic time series by applying jittering
    # synthetic_experts += apply_jittering_to_dataset(synthetic_experts, num_jitters_per_sample=10)
    #
    # # Add synthetic experts into original expert set
    # train_set[1] += synthetic_experts
    #
    # print("Number of training set experts after data augmentation:", len(train_set[1]))

    # ---------------------------------------- #


    # Convert all time series into the structure required for the model
    # Note: sets are 2 element tuples corresponding to the two skill levels, where each element is a list storing
    # multiple time series of some skill level

    train_set, val_set, test_set = prepare_sets_for_model(train_set, val_set, test_set)

    # ------------ DEBUGGING -------------- #

    print("Before window slicing....")
    print("\nNumber of novices and experts in training set: ")
    print("Novices:", len(train_set[0]))
    print("Experts:", len(train_set[1]))
    print("Number of novices and experts in validation set: ")
    print("Novices:", len(val_set[0]))
    print("Experts:", len(val_set[1]))
    print("Number of novices and experts in testing set: ")
    print("Novices:", len(test_set[0]))
    print("Experts:", len(test_set[1]))
    # -------------------------------------- #

    train_set, val_set, test_set = ut.prepare_folds(train_set, val_set, test_set, SLICE_WINDOW)

    x_train, y_train = train_set
    x_val, y_val = val_set
    x_test, y_test = test_set

    # ------------ DEBUGGING -------------- #
    print("\nAfter window slicing...\n")
    print("Number of novices, and experts in training set: ")
    num_novices = 0
    num_experts = 0
    for y in y_train:
        if y == 0:
            num_novices += 1
        else:
            num_experts += 1
    print("Novices:", num_novices)
    print("Experts:", num_experts)
    print("\nNumber of novices, and experts in validation set: ")
    num_novices = 0
    num_experts = 0
    for y in y_val:
        if y == 0:
            num_novices += 1
        else:
            num_experts += 1
    print("Novices:", num_novices)
    print("Experts:", num_experts)
    print("\nNumber of novices and experts in test set: ")
    num_novices = 0
    num_experts = 0
    for y in y_test:
        if y == 0:
            num_novices += 1
        else:
            num_experts += 1
    print("Novices:", num_novices)
    print("Experts:", num_experts)

    # -------------------------- #

    model = build_compile_and_fit_model(HYPER_PARAMETERS, train_set, val_set)

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print('\nTest accuracy', test_acc)
    print('Test loss', test_loss)

    print("\nMaking predictions on test set...")
    y_pred = model.predict(x_test)
    # print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)  # changes predictions to be single outputs, 0 or 1

    # Issue: all predictions are experts...

    # ---------- DEBUGGING ----------- #
    print("\nAfter making predictions...")
    print("\nTest set size:", len(y_test))
    print("\nPredictions size:", len(y_pred))
    num_pred_novices = 0
    num_pred_experts = 0
    for pred in y_pred:
        if pred == 1:
            num_pred_experts += 1
        if pred == 0:
            num_pred_novices += 1
    print("\nNumber of predicted novices:", num_pred_novices)
    print("Number of predicted experts:", num_pred_experts)

    num_test_novices = 0
    num_test_experts = 0
    for pred in y_test:
        if pred == 1:
            num_test_experts += 1
        if pred == 0:
            num_test_novices += 1
    print("\nNumber of novices in test set:", num_test_novices)
    print("Number of experts in test set:", num_test_experts)
    # ---------------------------------#

    print("\nF1 score on test set:")
    f_score = f1_score(y_true=y_test, y_pred=y_pred)
    print("F1 score:", f_score)


main()
