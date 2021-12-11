#%%
from warnings import simplefilter
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.layers.wrappers import Bidirectional
from torch._C import BenchmarkConfig
import torch.nn as nn
import joblib
from preprocessing_helpers import *
from data_collecting import hashtags
from tensorflow import keras
import os
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tf_hyperparameter_tuning import get_compiled_model


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
bucket.download_file(
    "artefacts/model_just_realdata.hdf5", "../artefacts/model_just_realdata.hdf5"
)

#%%
FINAL_PARAMS = {
    "embedding_dim": 64,
    "hidden_size": 48,
    "hidden_dense_dim": 112,
    "dropout_rate": 0.25,
    "l2_reg": 2.081445e-08,
}
model = get_compiled_model(**FINAL_PARAMS, learning_rate=0.001)

model.load_weights("../artefacts/model_just_realdata.hdf5")

#%%
embeddings = model.get_weights()[0]  # embeddings

#%%
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(embeddings)

#%%
def get_most_similar_words(n: int, word: str, similarity_matrix: np.ndarray):
    """
    Returns the n most similar words to the given word.
    """
    word_idx = encoder.encode(word)
    word_vec = embeddings[word_idx]
    similarities = similarity_matrix[word_idx]
    most_similar_idxs = similarities.argsort()[-n:][::-1]
    most_similar_words = [encoder.decode([idx]) for idx in most_similar_idxs]
    return most_similar_words


get_most_similar_words(10, "barcelona", similarity_matrix)

#%%
picked_words = [
    "hamilton",  # formula1
    "barcelona",  # championsleague
    "vaccine",  # covid19
    "christmas",  # holidays
    "turkey",  # thanksgiving
    "elon",  # tesla
    "btc",  # crypto
]

output_df = pd.DataFrame(
    {word: ", ".join(get_most_similar_words(10, word, similarity_matrix)) for word in picked_words},
    # columns=["word", "most_similar"],
)

output_df
#%%
from sklearn.manifold import TSNE

tsne = TSNE(n_jobs=-1)
tsne_embeddings = tsne.fit_transform(embeddings)

#%%
tsne_df = pd.DataFrame(tsne_embeddings, columns=["x", "y"])
