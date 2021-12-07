#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder

with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)

df = pd.read_parquet("../data/clean_tweets.parquet")
encoded_tweets = [encoder.encode(tweet) for tweet in df.tweet]
encoded_tweets = [" ".join(str(x) for x in encoded_tweets[idx].numpy()) for idx in range(len(encoded_tweets))]


#%%
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
X = cv.fit_transform(encoded_tweets)