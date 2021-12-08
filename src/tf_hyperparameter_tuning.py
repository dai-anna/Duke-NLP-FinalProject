#%%
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.wrappers import Bidirectional
import torch
import torch.nn as nn
import joblib
from preprocessing_helpers import *
from data_collecting import hashtags
from tensorflow import keras


#%%
# Load data from disk
encoder = joblib.load("../artefacts/encoder.pickle")

train = pd.read_parquet("../data/train.parquet")
val = pd.read_parquet("../data/val.parquet")
test = pd.read_parquet("../data/test.parquet")

xtrain, ytrain = encode_dataframe(encoder, data=train, mode="pytorch")
xval, yval = encode_dataframe(encoder, data=val, mode="pytorch")
xtest, ytest = encode_dataframe(encoder, data=test, mode="pytorch")

# Pad my input sequence with zeros
xtrain = nn.utils.rnn.pad_sequence(sequences=xtrain, batch_first=True, padding_value=0.0)
xval = nn.utils.rnn.pad_sequence(sequences=xval, batch_first=True, padding_value=0.0)
xtest = nn.utils.rnn.pad_sequence(sequences=xtest, batch_first=True, padding_value=0.0)

#%%

#%%
BATCH_SIZE = 64
LEARNING_RATE = 10 ** -2.5
NUM_EPOCHS = 20


def one_training_run(params: dict):
    # --------------------- Param & Data setup ---------------------#
    EMBEDDING_DIM = params["embedding_dim"]
    HIDDEN_SIZE = params["hidden_size"]
    HIDDEN_DENSE_DIM = params["hidden_dense_dim"]
    DROPOUT_RATE = params["dropout_rate"]
    L2_REG = params["l2_reg"]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (xtrain, ytrain.cat.codes.values)
    ).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval.cat.codes.values)).batch(
        BATCH_SIZE
    )

    # --------------------- Define the model ---------------------#
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                input_dim=encoder.vocab_size, output_dim=EMBEDDING_DIM
            ),
            keras.layers.Bidirectional(
                keras.layers.GRU(
                    HIDDEN_SIZE,
                    return_sequences=False,
                    kernel_regularizer=keras.regularizers.l2(L2_REG),
                )
            ),
            keras.layers.Dropout(rate=DROPOUT_RATE),
            keras.layers.Dense(
                units=HIDDEN_DENSE_DIM,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(L2_REG),
            ),
            keras.layers.Dropout(rate=DROPOUT_RATE),
            keras.layers.Dense(
                7,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(L2_REG),
            ),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )

    # --------------------- Define Checkpoints ---------------------#
    # model_checkpoint_loss = tf.keras.callbacks.ModelCheckpoint(
    #     filepath="Duke-NLP-FinalProject/data/trained_model/by_accuracy/",
    #     monitor="val_loss",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     mode="min",
    #     save_freq="epoch",
    # )

    # model_checkpoint_acc = tf.keras.callbacks.ModelCheckpoint(
    #     filepath="Duke-NLP-FinalProject/data/trained_model/by_loss/",
    #     monitor="val_accuracy",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     mode="max",
    #     save_freq="epoch",
    # )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    load_best_path = "../data/trained_model/by_accuracy/"

    # --------------------- Fit the model ---------------------#
    hist = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        # callbacks=[model_checkpoint_loss, model_checkpoint_acc],
        callbacks=[early_stopping_cb],
    )

    # model.load_weights(load_best_path)

    val_loss, val_accuracy = model.evaluate(val_dataset)

    return val_accuracy


def objective(trial):
    # --------------------- Search Space Definition ---------------------#
    embedding_dim = 2 ** trial.suggest_int("embedding_dim", 4, 6)
    hidden_size = 2 ** trial.suggest_int("hidden_size", 4, 8)
    hidden_dense_dim = 2 ** trial.suggest_int("hidden_dense_dim", 4, 8)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 0.000000001, 0.5, log=True)

    # --------------------- Run ---------------------#
    val_acc = one_training_run(
        params={
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "hidden_dense_dim": hidden_dense_dim,
            "dropout_rate": dropout_rate,
            "l2_reg": l2_reg,
        }
    )

    return val_acc


#%%
# --------------------- Setup Optuna ---------------------#

CREATE_NEW_STUDY = True

if CREATE_NEW_STUDY:
    study = optuna.create_study(
        "sqlite:///../data/tf_hyperparameter_study.db",
        direction="maximize",
        study_name="tf_study001",
    )
else:
    study = optuna.load_study(
        "no-name-40d6e161-1892-4aa4-a5f1-22029bb1507e",
        storage="sqlite:///../data/tf_hyperparameter_study.db",
    )

study.optimize(objective, n_trials=50)  # start study
print("-" * 80)
print(f"Found best params {study.best_params}")


# %%
from optuna.visualization import plot_parallel_coordinate

plot_parallel_coordinate(study)

# %%
