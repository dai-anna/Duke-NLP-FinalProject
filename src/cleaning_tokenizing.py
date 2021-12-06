#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder
from data_collecting import hashtags
from string import punctuation

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
    data["tweet"] = data["tweet"].str.replace(r"@[A-Za-z0-9_]+ ", "", regex=True)
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
    """ To lowercase + strip punctuation """
    data["tweet"] = data["tweet"].str.lower()
    data["tweet"] = data["tweet"].str.replace(r"""[!"#\$%&'\(\)\*\+,-\./:;\<=\>?\[\]\^_`\{\|\}~]""", " ")
    return data

def whitespaceencoder(data: pd.DataFrame) -> pd.DataFrame:
    input = data['tweet'].tolist()
    encoder = WhitespaceEncoder(input)
    encoded_data = [encoder.encode(example) for example in input]
    with open("../artefacts/encoder.pickle", "wb") as file:
        joblib.dump(encoder, file)
        
    return encoded_data
    

clean_df = (
    df.copy()
    .pipe(filter_tweets)
    .pipe(remove_usernames)
    .pipe(remove_urls)
    .pipe(remove_hashtags_and_cashtags)
)

#%%
# convert tweet column to string and hashtag to category
clean_df.tweet = clean_df.tweet.astype("string")
clean_df.hashtag = clean_df.hashtag.astype("category")


#%%
clean_df.info()
clean_df.to_parquet("../data/clean_tweets.parquet")
#%%
# for x in clean_df.tweet:
#     print(x)
