#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder

with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)

df = pd.read_parquet("../data/clean_tweets.parquet")

#%%
df

#%%
encoded = [encoder.encode(tweet) for tweet in df.tweet]
encoded[0]
#%%
[encoder.decode(x) for x in encoded]

#%%
print(encoder.vocab_size)


#%%
with open("dump.txt", "w") as f:
    for v in encoder.vocab:
        f.write(v + "\n")