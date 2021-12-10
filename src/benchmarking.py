#%%
from numpy.core.numeric import False_
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.wrappers import Bidirectional
import torch
import torch.nn as nn
import joblib
from preprocessing_helpers import *
from data_collecting import hashtags
from tensorflow import keras
import os

#%%
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

synth_xtrain, synth_ytrain = encode_dataframe(encoder, data=synth_train, mode="pytorch")
synth_xval, synth_yval = encode_dataframe(encoder, data=synth_val, mode="pytorch")
synth_xtest, synth_ytest = encode_dataframe(encoder, data=synth_test, mode="pytorch")

# Pad my input sequence with zeros
synth_xtrain = nn.utils.rnn.pad_sequence(
    sequences=synth_xtrain, batch_first=True, padding_value=0.0
)
synth_xval = nn.utils.rnn.pad_sequence(
    sequences=synth_xval, batch_first=True, padding_value=0.0
)
synth_xtest = nn.utils.rnn.pad_sequence(
    sequences=synth_xtest, batch_first=True, padding_value=0.0
)


#%%
from tf_hyperparameter_tuning import get_compiled_model

BATCH_SIZE = 64
LEARNING_RATE = 10 ** -2.5
NUM_EPOCHS = 20
FINAL_PARAMS = {
    "embedding_dim": 2 ** 5,
    "hidden_size": 2 ** 6,
    "hidden_dense_dim": 2 ** 6,
    "dropout_rate": 0.1,
    "l2_reg": 0,
}

model = get_compiled_model(**FINAL_PARAMS, learning_rate=LEARNING_RATE)
bucket.download_file(
    "artefacts/model_synthdata.hdf5", "../artefacts/model_synthdata.hdf5"
)
model.load_weights("../artefacts/model_synthdata.hdf5")

#%%
raw_preds = model.predict(synth_xtest.numpy())

#%%
preds = raw_preds.argmax(axis=1)

#%%
from sklearn.metrics import classification_report

lda_topic_real_topic_mapper = {
    "0": "thanksgiving",  # bad, mixed with holidays, stock market, crypto
    "1": "formula1",  # very good
    "2": "covid",  # => general stock market?
    "3": "championsleague",  # + covid19
    "4": "crypto",  # good
    "5": "tesla",  # good
    "6": "holidays",  # + covid
}


synth_model_df = (
    pd.DataFrame(
        classification_report(preds, synth_ytest.cat.codes.values, output_dict=True)
    )[[str(x) for x in range(7)]]
    .rename(lda_topic_real_topic_mapper, axis=1)
    .round(3)
    .T.assign(support=lambda d: d["support"].astype("int"))
)
synth_model_df


#%%
with open(
    "../report/benchmark_outputs/synth_only_model_classificationreport.tex", "w"
) as f:
    synth_model_df.to_latex(buf=f, escape=False, index=True)
