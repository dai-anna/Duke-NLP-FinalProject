#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder
from preprocessing_helpers import encode_dataframe
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from data_collecting import hashtags
import numpy as np
import os
from sklearn.model_selection import train_test_split

np.random.seed(42)

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
# ----------------------------------- LOAD FROM DISK -----------------------------------
# load encoder
with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)

# load data
train = pd.read_parquet("../data/train.parquet")
val = pd.read_parquet("../data/val.parquet")
test = pd.read_parquet("../data/test.parquet")

# for LDA: train on train + val as there are no HP
train = pd.concat([train, val])

xtrain, ytrain = train["tweet"], train["hashtag"]
xtest, yttest = test["tweet"], test["hashtag"]

for x_ in (xtrain, xtest):
    print(len(x_))

#%%
# ----------------------------------- Vectorize data and fit model -----------------------------------
cv = CountVectorizer(vocabulary=encoder.token_to_index)

xtrain_matrix = cv.transform(xtrain)
xtest_matrix = cv.transform(xtest)

RETRAIN = False

if RETRAIN:
    lda = LatentDirichletAllocation(n_components=7, random_state=42, n_jobs=-1)
    lda.fit(xtrain_matrix)

    # Save to disk
    joblib.dump(cv, "../artefacts/lda_vectorizer.joblib")
    joblib.dump(lda, "../artefacts/lda_model.joblib")

    # Save to S3
    if SAVE_TO_S3:
        bucket.upload_file(
            "../artefacts/lda_vectorizer.joblib", "artefacts/lda_vectorizer.joblib"
        )
        bucket.upload_file("../artefacts/lda_model.joblib", "artefacts/lda_model.joblib")
        print("[INFO] LDA model and vectorizer saved to S3.")

else:
    cv = joblib.load("../artefacts/lda_vectorizer.joblib")
    lda = joblib.load("../artefacts/lda_model.joblib")

#%%
# ----------------------------------- Print top words per topic -----------------------------------
top_k_per_topic = lda.components_.argsort(axis=1)[:, -50:]
for idx, topic in enumerate(top_k_per_topic):
    print("=" * 20 + f"Topic #{idx}" + "=" * 20)
    print(encoder.decode(topic[::-1]))
    print()


#%%
# ----------------------------------- Sample words from one topic -----------------------------------
def sample_from_topic(topic_idx: int, n_samples: int):
    comp = lda.components_[topic_idx, :]
    comp = comp / comp.sum()

    return encoder.decode(
        np.random.choice(np.arange(encoder.vocab_size), p=comp, size=n_samples)
    )


[sample_from_topic(6, 20) for _ in range(10)]

#%%

# Original topics: crypto, tesla, championsleague, formula1, thanksgiving, holidays, covid19
lda_topic_real_topic_mapper = {
    0: "thanksgiving",  # bad, mixed with holidays, stock market, crypto
    1: "formula1",  # very good
    2: "covid",  # => general stock market?
    3: "championsleague",  # + covid19
    4: "crypto",  # good
    5: "tesla",  # good
    6: "holidays",  # + covid
}

# worst overlap: tesla & thanksgiving => stock market, covid19 & everything


#%%
# ----------------------------------- Generate Synthetic Data -----------------------------------
GENERATE_DATA = False


def generate_synthetic_data(n_samples_per_topic: int):
    tweet_length_distribution = train["tweet"].apply(lambda r: len(r.split())).values

    synth_data = []
    for idx, hashtag in enumerate(hashtags):
        for _ in range(n_samples_per_topic):
            synth_data.append(
                (
                    sample_from_topic(
                        topic_idx=idx,
                        n_samples=np.random.choice(tweet_length_distribution),
                    ),
                    lda_topic_real_topic_mapper[idx],
                )
            )

    return synth_data


if GENERATE_DATA:
    synth_df = pd.DataFrame(
        generate_synthetic_data(n_samples_per_topic=10_000),
        columns=["tweet", "hashtag"],
    )
    synth_df["hashtag"] = synth_df["hashtag"].astype("category")

    # ----------------------------------- Save Synthetic Data -----------------------------------
    synth_df.to_parquet("../data/synth_data.parquet")

    xtrain, xval, ytrain, yval = train_test_split(
        synth_df["tweet"].to_frame(),
        synth_df["hashtag"],
        test_size=0.4,
        random_state=42,
    )

    xval, xtest, yval, ytest = train_test_split(
        xval, yval, test_size=0.5, random_state=42
    )

    for x_ in (xtrain, xval, xtest):
        print(x_.shape)

    pd.concat([xtrain, ytrain], axis=1).to_parquet("../data/synth_train.parquet")
    pd.concat([xval, yval], axis=1).to_parquet("../data/synth_val.parquet")
    pd.concat([xtest, ytest], axis=1).to_parquet("../data/synth_test.parquet")
    print("Saved synth parquets to disk.")

    if SAVE_TO_S3:
        bucket.upload_file("../data/synth_data.parquet", "data/synth_data.parquet")
        bucket.upload_file("../data/synth_train.parquet", "data/synth_train.parquet")
        bucket.upload_file("../data/synth_val.parquet", "data/synth_val.parquet")
        bucket.upload_file("../data/synth_test.parquet", "data/synth_test.parquet")
        print("[INFO] Synthetic data saved to S3.")


#%%
EXPORT_EXAMPLE_TABLE = True

if EXPORT_EXAMPLE_TABLE:
    _df = pd.DataFrame(
        {
            "Topic": ["Formula1", "Crypto", "Tesla"],
            "Sampled": [
                sample_from_topic(1, 10),
                sample_from_topic(4, 10),
                sample_from_topic(2, 10),
            ],
        }
    ).set_index("Topic")
    pd.options.display.max_colwidth = 100
    print(_df)
    _df.to_latex("../report/lda_samples.tex", index=True)