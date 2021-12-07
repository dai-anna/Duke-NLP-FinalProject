#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder
from preprocessing_helpers import encode_dataframe
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# ----------------------- LOAD FROM DISK -----------------------
# load encoder
with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)

# load data
train = pd.read_parquet("../data/train.parquet")
val = pd.read_parquet("../data/val.parquet")
test = pd.read_parquet("../data/test.parquet")

# for LDA: train on train + val as there are no HP
train = pd.concat([train, val])


# xtrain, ytrain = encode_dataframe(encoder, data=train, mode="sklearn")
# xval, yval = encode_dataframe(encoder, data=val, mode="sklearn")
# xtest, ytest = encode_dataframe(encoder, data=test, mode="sklearn")

xtrain, ytrain = train["tweet"], train["hashtag"]
xtest, yttest = test["tweet"], test["hashtag"]

for x_ in (xtrain, xtest):
    print(len(x_))

#%%

cv = CountVectorizer(vocabulary=encoder.token_to_index)
xtrain_matrix = cv.transform(xtrain)
xtest_matrix = cv.transform(xtest)


#%%

lda = LatentDirichletAllocation(n_components=7, random_state=42, n_jobs=-1)
lda.fit(xtrain_matrix)

#%%
lda.transform(xtest_matrix)


#%%
top_k_per_topic = lda.components_.argsort(axis=1)[:, -30:]
for idx, topic in enumerate(top_k_per_topic):
    print("=" * 20 + f"Topic #{idx}" + "=" * 20)
    print(encoder.decode(topic))
    print()


#%%
comp = lda.components_[0, :]
comp = comp / comp.sum()

encoder.decode(np.random.choice(np.arange(encoder.vocab_size), p=comp, size=10))

#%%
lda.components_