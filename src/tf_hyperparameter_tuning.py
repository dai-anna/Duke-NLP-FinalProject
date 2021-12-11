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
import os

try:
    import boto3

    s3 = boto3.resource(
        "s3",
        region_name="us-east-1",
        aws_secret_access_key=os.getenv("AWS_SECRET_AK"),
        aws_access_key_id=os.getenv("AWS_AK"),
    )
    bucket = s3.Bucket("ids703-nlp-finalproject")
    SAVE_TO_S3 = True
    print("[INFO] S3 connection successful.")
except:
    print("[ERROR] Could not connect to S3! Only saving locally.")
    SAVE_TO_S3 = False


#%%
# Load data from disk
USE_SYNTHETIC_DATA = False

encoder = joblib.load("../artefacts/encoder.pickle")

train = pd.read_parquet(f"../data/{'synth_' if USE_SYNTHETIC_DATA else ''}train.parquet")
val = pd.read_parquet(f"../data/{'synth_' if USE_SYNTHETIC_DATA else ''}val.parquet")
test = pd.read_parquet(f"../data/{'synth_' if USE_SYNTHETIC_DATA else ''}test.parquet")

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


def get_compiled_model(
    embedding_dim, hidden_size, hidden_dense_dim, dropout_rate, l2_reg, learning_rate
):
    # --------------------- Define the model ---------------------#
    model = keras.Sequential(
        [
            keras.layers.Embedding(
                input_dim=encoder.vocab_size, output_dim=embedding_dim
            ),
            keras.layers.Bidirectional(
                keras.layers.GRU(
                    hidden_size,
                    return_sequences=False,
                    kernel_regularizer=keras.regularizers.l2(l2_reg),
                )
            ),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(
                units=hidden_dense_dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
            keras.layers.Dropout(rate=dropout_rate),
            keras.layers.Dense(
                7,
                activation="softmax",
                kernel_regularizer=keras.regularizers.l2(l2_reg),
            ),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )

    return model


def one_training_run(params: dict):
    # --------------------- Param & Data setup ---------------------#
    # EMBEDDING_DIM = params["embedding_dim"]
    # HIDDEN_SIZE = params["hidden_size"]
    # HIDDEN_DENSE_DIM = params["hidden_dense_dim"]
    # DROPOUT_RATE = params["dropout_rate"]
    # L2_REG = params["l2_reg"]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (xtrain, ytrain.cat.codes.values)
    ).batch(BATCH_SIZE)
    val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval.cat.codes.values)).batch(
        BATCH_SIZE
    )

    model = get_compiled_model(**params, learning_rate=LEARNING_RATE)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )

    # --------------------- Fit the model ---------------------#
    hist = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping_cb],
    )

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

if __name__ == "__main__":
    CREATE_NEW_STUDY = True
    if CREATE_NEW_STUDY:
        study = optuna.create_study(
            f"sqlite:///../artefacts/tf_hyperparameter_study_{'synth' if USE_SYNTHETIC_DATA else 'real'}.db",
            direction="maximize",
            study_name="tf_study001",
        )
    else:
        study = optuna.load_study(
            "tf_study001",
            storage=f"sqlite:///../artefacts/tf_hyperparameter_study_{'synth' if USE_SYNTHETIC_DATA else 'real'}.db",
        )

    study.optimize(objective, n_trials=50)  # start study
    print("-" * 80)
    print(f"Found best params {study.best_params}")

    # save study to s3
    if SAVE_TO_S3:
        bucket.upload_file(
            f"../artefacts/tf_hyperparameter_study_{'synth' if USE_SYNTHETIC_DATA else 'real'}.db",
            f"artefacts/tf_hyperparameter_study_{'synth' if USE_SYNTHETIC_DATA else 'real'}.db",
        )
    print("[INFO] Study saved to S3")
