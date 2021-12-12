#%%
import pandas as pd
import tensorflow as tf
import torch.nn as nn
import joblib
from preprocessing_helpers import encode_dataframe
import numpy as np
from tensorflow import keras
from tf_hyperparameter_tuning import get_compiled_model
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
test = pd.read_parquet(f"../data/{'synth_' if USE_SYNTHETIC_DATA else ''}test.parquet")
xtest, ytest = encode_dataframe(encoder, data=test, mode="pytorch")
xtest = nn.utils.rnn.pad_sequence(sequences=xtest, batch_first=True, padding_value=0.0)

#%%
FINAL_PARAMS = {
    "embedding_dim": 64,
    "hidden_size": 48,
    "hidden_dense_dim": 112,
    "dropout_rate": 0.25,
    "l2_reg": 2.081445e-08,
}

# load nn model
bucket.download_file(
    "artefacts/model_just_realdata.hdf5",
    "../artefacts/model_just_realdata.hdf5",
)
model = get_compiled_model(**FINAL_PARAMS, learning_rate=42)  # doesnt matter here
model.load_weights("../artefacts/model_just_realdata.hdf5")

bucket.download_file(
    "artefacts/lda_vectorizer.joblib", "../artefacts/lda_vectorizer.joblib"
)
bucket.download_file("artefacts/lda_model.joblib", "../artefacts/lda_model.joblib")

cv = joblib.load("../artefacts/lda_vectorizer.joblib")
lda = joblib.load("../artefacts/lda_model.joblib")

#%%
TWEET_IDX = 0
TWEET_TEXT = test.iloc[TWEET_IDX]['tweet']

#%%
nn_pred = model.predict(xtest[TWEET_IDX].numpy().reshape(1, -1))
lda_pred = lda.transform(cv.transform([TWEET_TEXT]))
pred_mtx = np.vstack((nn_pred, lda_pred))

#%%
fig, ax = plt.subplots(figsize=(10, 3))
sns.heatmap(np.round(pred_mtx, 3), square=True, annot=True, cmap="viridis", annot_kws=dict(size=16, weight="bold"))