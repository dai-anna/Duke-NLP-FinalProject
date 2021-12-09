#%%
import pandas as pd
from torchnlp.encoders.text import WhitespaceEncoder


def encode_dataframe(
    encoder: WhitespaceEncoder, data: pd.DataFrame, mode="pytorch"
) -> tuple[list, pd.Series]:
    """Encode a dataframe with a given encoder. Splits and returns X and y separately."""
    encoded_tweets = [encoder.encode(tweet) for tweet in data["tweet"]]

    if mode == "sklearn":
        encoded_tweets = [
            " ".join(str(x) for x in encoded_tweets[idx].numpy())
            for idx in range(len(encoded_tweets))
        ]

    return encoded_tweets, data["hashtag"]
