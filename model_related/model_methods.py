from model_related.mhadatareader import MhaDataReader
from model_related.classes import ParticipantsData, Scan, ProficiencyLabel
import model_related.utils as ut
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import random
from tensorflow.keras.utils import plot_model
from sklearn.metrics import recall_score, confusion_matrix
import seaborn as sn
import pandas as pd

import os


def save_model(model, fold):
    model.save(f'./model_{fold}.tf')


def build_model(input_shape, num_classes, filters, kernel_size, dropout_rate, regularizer):
    input_layer = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same",
                                kernel_regularizer=regularizer,
                                activation='relu')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Dropout(dropout_rate)(conv1)

    conv2 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                padding="same", kernel_regularizer=regularizer,
                                activation='relu')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Dropout(dropout_rate)(conv2)

    conv3 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size,
                                padding="same", kernel_regularizer=regularizer,
                                activation='relu')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Dropout(dropout_rate)(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


def load_model(num, reg):
    build = True

    m_name_tuned = f'./model_tuned_{num}_{reg}.tf'
    if os.path.exists(m_name_tuned):
        print(f'loading prev tuned {m_name_tuned}')
        return keras.models.load_model(m_name_tuned), not build

    m_name_untuned = f'./model_{num}.tf'
    print(f'loading non-tuned {m_name_untuned}')
    return keras.models.load_model(m_name_untuned), build


def save_model_tune(model, num, reg):
    m_name_tuned = f'./model_tuned_{num}_{reg}.tf'

    print(f'saving {m_name_tuned}')
    model.save(m_name_tuned)


def build_model_funetune(base_model, input_shape, num_classes, filters,
                         kernel_size, dropout_rate, regularizer):
    assert base_model is not None
    model = keras.Sequential()

    # freeze the base model
    base_model.trainable = False
    # add all layers except the last two
    for layer in base_model.layers[:-2]:
        model.add(layer)

    model.add(keras.layers.Conv1D(filters=filters,
                                  kernel_size=kernel_size, padding="same",
                                  kernel_regularizer=regularizer,
                                  activation='relu',
                                  name=f'Conv1D_{str(len(model.layers) + 1)}'))
    model.add(keras.layers.BatchNormalization(
        name=f'BatchNormalization_{str(len(model.layers) + 1)}'
    ))
    model.add(keras.layers.Dropout(dropout_rate,
                                   name=f'Dropout_{str(len(model.layers) + 1)}'
                                   ))

    model.add(keras.layers.GlobalAveragePooling1D(
        name=f'GlobalAveragePooling1D_{str(len(model.layers) + 1)}'))
    model.add(keras.layers.Dense(num_classes, activation='softmax',
                                 name=f'Dense_{str(len(model.layers) + 1)}'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

    return model


def build_compile_and_fit_model(hyperparameters: dict, train_set: tuple, val_set: tuple):

    x_train, y_train = train_set
    x_val, y_val = val_set
    CALLBACKS = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.000001),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            verbose=1,
        )
    ]
    optimizer = keras.optimizers.Adam(
        learning_rate=hyperparameters["learning-rate"],
    )
    model = build_model(
        x_train.shape[1:],
        len(ProficiencyLabel),
        kernel_size=hyperparameters["kernel-size"],
        filters=hyperparameters["filters"],
        dropout_rate=hyperparameters["dropout-rate"],
        regularizer=hyperparameters["regularizer"]
    )
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    model.fit(
        x_train,
        y_train,
        batch_size=hyperparameters["batch-size"],
        epochs=hyperparameters["epochs"],
        callbacks=CALLBACKS,
        validation_data=(x_val, y_val),
        verbose=1,
    )
    return model


