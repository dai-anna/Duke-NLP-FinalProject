#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.wrappers import Bidirectional
import torch.nn as nn
import joblib
from preprocessing_helpers import *
from data_collecting import hashtags
from tensorflow import keras
import os
from sklearn.metrics import classification_report

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

# synth data
synth_test = pd.read_parquet("../data/synth_test.parquet")
synth_xtest, synth_ytest = encode_dataframe(encoder, data=synth_test, mode="pytorch")
synth_xtest = nn.utils.rnn.pad_sequence(
    sequences=synth_xtest, batch_first=True, padding_value=0.0
)

# real data
test = pd.read_parquet("../data/test.parquet")
xtest, ytest = encode_dataframe(encoder, data=test, mode="pytorch")
xtest = nn.utils.rnn.pad_sequence(sequences=xtest, batch_first=True, padding_value=0.0)


#%%
# --------------------------- Loading Synth only Model ---------------------------
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

# --------------------------- Synth only model benchmarks ---------------------------
#%%
def do_benchmark_and_save(model, xtest, ytest, tex_output_path, caption):
    """Conducts benchmark on test set and saves results to tex file."""

    lda_topic_real_topic_mapper = {
        "0": "thanksgiving",  # bad, mixed with holidays, stock market, crypto
        "1": "formula1",  # very good
        "2": "covid",  # => general stock market?
        "3": "championsleague",  # + covid19
        "4": "crypto",  # good
        "5": "tesla",  # good
        "6": "holidays",  # + covid
    }

    raw_preds = model.predict(xtest.numpy())
    preds = raw_preds.argmax(axis=1)

    # collect results in dataframe
    benchmark_df = (
        pd.DataFrame(
            classification_report(preds, ytest.cat.codes.values, output_dict=True)
        )[[str(x) for x in range(7)]]
        .rename(lda_topic_real_topic_mapper, axis=1)
        .round(3)
        .T.assign(support=lambda d: d["support"].astype("int"))
    )

    # save
    with open(tex_output_path, "w") as f:
        benchmark_df.to_latex(
            buf=f,
            escape=False,
            index=True,
            bold_rows=False,
        )

    # Ghetto way to get the caption to appear under the table
    with open(tex_output_path, "r") as f:
        content = f.readlines()
        content.insert(-1, "\\caption{" + caption + "}\n")

    with open(tex_output_path, "w") as f:
        f.writelines(content)


#%%
do_benchmark_and_save(
    model,
    synth_xtest,
    synth_ytest,
    "../report/benchmark_outputs/synth_only_model_classificationreport_synthdata.tex",
    "Benchmark results of neural net (trained on synthetic data only) on synthetic data",
)

#%%
do_benchmark_and_save(
    model,
    xtest,
    ytest,
    "../report/benchmark_outputs/synth_only_model_classificationreport_realdata.tex",
    "Benchmark results of neural net (trained on synthetic data only) on real data",
)


#%%
# --------------------------- Loading Synth+real Model ---------------------------
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
    "artefacts/model_synthdata_and_realdata.hdf5",
    "../artefacts/model_synthdata_and_realdata.hdf5",
)
model.load_weights("../artefacts/model_synthdata_and_realdata.hdf5")

#%%
# --------------------------- Synth+real model benchmarks ---------------------------
do_benchmark_and_save(
    model,
    synth_xtest,
    synth_ytest,
    "../report/benchmark_outputs/synthandreal_model_classificationreport_synthdata.tex",
    "Benchmark results of neural net (trained on synthetic data and real data) on synthetic data",
)

#%%
do_benchmark_and_save(
    model,
    xtest,
    ytest,
    "../report/benchmark_outputs/synthandreal_model_classificationreport_realdata.tex",
    "Benchmark results of neural net (trained on synthetic data and real data) on real data",
)

#%%
# TODO: Benchmark model that's only been trained on real data
