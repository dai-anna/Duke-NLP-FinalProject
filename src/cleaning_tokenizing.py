#%%
import pandas as pd
from data_collecting import hashtags

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
