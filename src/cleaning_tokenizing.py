#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder
from data_collecting import hashtags
from string import punctuation
import re

EMOJI_REGEX = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "])"
)

#%%
# merge all csvs into one dataframe
csv_files = [
    pd.read_csv(f"../data/twint_output_{ht}.csv", usecols=["tweet", "language"]).assign(
        hashtag=f"{ht}"
    )
    for ht in hashtags
]
df = pd.concat(csv_files)

#%%
def filter_tweets(data: pd.DataFrame) -> pd.DataFrame:
    data = data.query("language == 'en'").drop(columns=["language"])
    return data


def remove_usernames(data: pd.DataFrame) -> pd.DataFrame:
    data["tweet"] = data["tweet"].str.replace(r"@[A-Za-z0-9_]+\b", "", regex=True)
    return data


def remove_urls(data: pd.DataFrame) -> pd.DataFrame:
    data["tweet"] = data["tweet"].str.replace(
        r"https:\/\/t\.co\/[A-Za-z\d]+", "", regex=True
    )
    return data


def remove_hashtags_and_cashtags(data: pd.DataFrame) -> pd.DataFrame:
    data["tweet"] = data["tweet"].str.replace(r"#[A-Za-z0-9_]+\b", "", regex=True)
    data["tweet"] = data["tweet"].str.replace(r"\$[A-Za-z0-9_]+\b", "", regex=True)
    return data


def normalize_text(data: pd.DataFrame) -> pd.DataFrame:
    """To lowercase + strip punctuation + replace numbers"""
    data["tweet"] = data["tweet"].str.lower()
    data["tweet"] = data["tweet"].str.replace(
        r"""[!"#\$%&'\(\)\*\+,-\./:;\<=\>?\[\]\^_`\{\|\}~“”’]""", " ", regex=True
    )
    data["tweet"] = data["tweet"].str.replace(r"\d+", " <number> ", regex=True)
    return data


def space_out_emojis(data: pd.DataFrame) -> pd.DataFrame:
    data["tweet"] = data["tweet"].str.replace(
        EMOJI_REGEX, r" \1 ", regex=True
    )  # space before and after
    return data


def remove_multi_spaces(data: pd.DataFrame) -> pd.DataFrame:
    data["tweet"] = data["tweet"].str.replace(r"\s+", " ", regex=True).str.strip()
    return data


def whitespace_encode(data: pd.DataFrame) -> pd.DataFrame:
    input_ = data["tweet"].tolist()
    encoder = WhitespaceEncoder(input_, min_occurrences=2)
    encoded_data = [encoder.encode(example) for example in input_]
    with open("../artefacts/encoder.pickle", "wb") as file:
        joblib.dump(encoder, file)
    print("Saved encoder to disk.")


clean_df = (
    df.copy()
    .pipe(filter_tweets)
    .pipe(remove_usernames)
    .pipe(remove_urls)
    .pipe(remove_hashtags_and_cashtags)
    .pipe(normalize_text)
    .pipe(space_out_emojis)
    .pipe(remove_multi_spaces)
)

#%%
# convert tweet column to string and hashtag to category
clean_df.tweet = clean_df.tweet.astype("string")
clean_df.hashtag = clean_df.hashtag.astype("category")


#%%
clean_df.info()
clean_df.to_parquet("../data/clean_tweets.parquet")

#%%
_ = whitespace_encode(clean_df)
