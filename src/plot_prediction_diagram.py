#%%
from numpy.lib.utils import byte_bounds
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
import matplotlib.image as mpimg
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
lda_topic_real_topic_mapper = {
    "0": "thanksgiving",  # bad, mixed with holidays, stock market, crypto
    "1": "formula1",  # very good
    "2": "covid",  # => general stock market?
    "3": "championsleague",  # + covid19
    "4": "crypto",  # good
    "5": "tesla",  # good
    "6": "holidays",  # + covid
}


nn_to_lda_topic_mapper = {
    0: 3,
    1: 2,
    2: 4,
    3: 1,
    4: 6,
    5: 5,
    6: 0,
}

inverse_nn_to_lda_topic_mapper = {v: k for k, v in nn_to_lda_topic_mapper.items()}
#%%
BAD_TWEET_IDX = 4030
GOOD_TWEET_IDX = 0

print(f"{' BAD ':=^80}")
print(f"#{BAD_TWEET_IDX:05d}: {test.iloc[BAD_TWEET_IDX]['tweet']}")
print(f"{' GOOD ':=^80}")
print(f"#{GOOD_TWEET_IDX:05d}: {test.iloc[GOOD_TWEET_IDX]['tweet']}")

""" 
===================================== BAD ======================================
#04030: maximalist ai ml tech etc obsoleting jobs – bitcoin fixes bitcoin standard going radically change system – adapt get left behind
===================================== GOOD =====================================
#00000: know makes great holiday gifts books course got paperbacks delight voracious reader life visit amazon page peruse books pick paperback today
"""


#%%

fig, axes = plt.subplots(2, 2, figsize=(20, 5))


def plot_colored_predictions(ax, tweet_idx):
    tweet_text = test.iloc[tweet_idx]["tweet"]
    nn_pred = model.predict(xtest[tweet_idx].numpy().reshape(1, -1))
    nn_pred = np.array(
        [nn_pred.ravel()[inverse_nn_to_lda_topic_mapper[p]] for p in range(7)]
    )
    lda_pred = lda.transform(cv.transform([tweet_text]))
    pred_mtx = np.vstack((nn_pred, lda_pred))

    sns.heatmap(
        np.round(pred_mtx, 3),
        square=True,
        annot=True,
        cmap="coolwarm",
        annot_kws=dict(size=16, weight="bold"),
        cbar=False,
        ax=ax,
    )

    topiclabels = [
        "Thanks-\ngiving",
        "Formula1",
        "Covid19",
        "Champions-\nleague",
        "Crypto",
        "Tesla",
        "Holidays",
    ]

    ax.set_yticklabels(["NN", "LDA"], size=16, weight="bold")
    ax.set_xticklabels(topiclabels, size=14, weight="normal")


axes[0, 0].imshow(mpimg.imread("../report/colored_predictions.png"))
axes[0, 0].axis("off")
axes[0, 1].imshow(mpimg.imread("../report/colored_predictions.png"))
axes[0, 1].axis("off")


plot_colored_predictions(axes[1][0], GOOD_TWEET_IDX)
plot_colored_predictions(axes[1][1], BAD_TWEET_IDX)
plt.tight_layout()
fig.savefig("../report/colored_predictions.png", facecolor="w", dpi=300, bbox_inches="tight")