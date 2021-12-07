#%%
from os import WNOWAIT
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder
from preprocessing_helpers import encode_dataframe

# ----------------------- LOAD FROM DISK -----------------------
# load encoder
with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)

# load data
train = pd.read_parquet("../data/train.parquet")
val = pd.read_parquet("../data/val.parquet")
test = pd.read_parquet("../data/test.parquet")

xtrain, ytrain = encode_dataframe(encoder, data=train, mode="sklearn")
xval, yval = encode_dataframe(encoder, data=val, mode="sklearn")
xtest, ytest = encode_dataframe(encoder, data=test, mode="sklearn")


#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit(xtrain)
