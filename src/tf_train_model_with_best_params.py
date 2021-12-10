#%%
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
encoder = joblib.load("../artefacts/encoder.pickle")

synth_train = pd.read_parquet("../data/synth_train.parquet")
synth_val = pd.read_parquet("../data/synth_val.parquet")
synth_test = pd.read_parquet("../data/synth_test.parquet")

xtrain, ytrain = encode_dataframe(encoder, data=synth_train, mode="pytorch")
xval, yval = encode_dataframe(encoder, data=synth_val, mode="pytorch")
xtest, ytest = encode_dataframe(encoder, data=synth_test, mode="pytorch")

# Pad my input sequence with zeros
xtrain = nn.utils.rnn.pad_sequence(sequences=xtrain, batch_first=True, padding_value=0.0)
xval = nn.utils.rnn.pad_sequence(sequences=xval, batch_first=True, padding_value=0.0)
xtest = nn.utils.rnn.pad_sequence(sequences=xtest, batch_first=True, padding_value=0.0)

#%%
BATCH_SIZE = 64
LEARNING_RATE = 10 ** -2.5
NUM_EPOCHS = 20


#%%
from tf_hyperparameter_tuning import get_compiled_model

FINAL_PARAMS = {
    "embedding_dim": 2 ** 5,
    "hidden_size": 2 ** 6,
    "hidden_dense_dim": 2 ** 6,
    "dropout_rate": 0.1,
    "l2_reg": 0,
}

model = get_compiled_model(**FINAL_PARAMS, learning_rate=LEARNING_RATE)

#%%
# ----------------------------------------- Synthetic Data -----------------------------------------
synth_train_dataset = tf.data.Dataset.from_tensor_slices(
    (xtrain, ytrain.cat.codes.values)
).batch(BATCH_SIZE)
synth_val_dataset = tf.data.Dataset.from_tensor_slices(
    (xval, yval.cat.codes.values)
).batch(BATCH_SIZE)

#%%
# Decide on learning rate based on plot HERE:
LEARNING_RATE = 10 ** -2.5
model = get_compiled_model(**FINAL_PARAMS, learning_rate=LEARNING_RATE)

#%%
# train model on synth data
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

# --------------------- Fit the model ---------------------#
REFIT_MODEL = False

if REFIT_MODEL:
    hist = model.fit(
        synth_train_dataset,
        validation_data=synth_val_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping_cb],
    )

    val_loss, val_accuracy = model.evaluate(synth_val_dataset)
    print(f"[INFO] Best validation acc = {val_accuracy}")
    # pd.DataFrame(hist.history)[["loss", "val_loss"]].plot(figsize=(8, 5))

    model.save_weights("../artefacts/model_synthdata.hdf5")

    if SAVE_TO_S3:
        bucket.upload_file(
            "../artefacts/model_synthdata.hdf5", "artefacts/model_synthdata.hdf5"
        )
    print("[INFO] Model saved to disk & uploaded to S3.")
else:
    bucket.download_file(
        "artefacts/model_synthdata.hdf5", "../artefacts/model_synthdata.hdf5"
    )
    model.load_weights("../artefacts/model_synthdata.hdf5")

#%%
# ----------------------------- Continue on Real Data -----------------------------
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

train_dataset = tf.data.Dataset.from_tensor_slices(
    (xtrain, ytrain.cat.codes.values)
).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval.cat.codes.values)).batch(
    BATCH_SIZE
)


hist = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping_cb],
)

val_loss, val_accuracy = model.evaluate(val_dataset)
print(f"[INFO] Best validation acc = {val_accuracy}")
# pd.DataFrame(hist.history)[["loss", "val_loss"]].plot(figsize=(8, 5))

model.save_weights("../artefacts/model_synthdata_and_realdata.hdf5")

if SAVE_TO_S3:
    bucket.upload_file(
        "../artefacts/model_synthdata_and_realdata.hdf5", "artefacts/model_synthdata_and_realdata.hdf5"
    )
print("[INFO] Model saved to disk & uploaded to S3.")
