#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder

df = pd.read_csv("../data/twint_output_crypto.csv")

#%%
def filter_tweets(data: pd.DataFrame) -> pd.DataFrame:
    return data.query("language == 'en'")


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
clean_df



#%%
for x in clean_df.tweet:
    print(x)
