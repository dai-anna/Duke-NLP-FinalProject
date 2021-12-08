import twint
import requests
import os
import pandas as pd
import time
import joblib

hashtags = [
    "crypto",
    "tesla",
    "championsleague",
    "formula1",
    "thanksgiving",
    "holidays",
    "covid19",
]


def scrape_tweets_from_hashtags(hashtags):
    """Scrape tweets from hashtags using Twint"""

    for _, trend in enumerate(hashtags):
        c = twint.Config()
        c.Search = "#" + trend  # your search here
        c.Lang = "en"
        c.Limit = 10_000
        c.Store_csv = True
        c.Output = f"data/twint_output_{trend}.csv"
        twint.run.Search(c)


def pull_tweets_for_hashtag(hashtag):
    """Pull tweets using Twitter API"""

    result_list = []

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {os.environ['TWITTER_BEARER']}",
    }

    params = {
        "query": f"#{hashtag} lang:en -is:retweet",
        "max_results": 100,  # min: 10, max: 100
    }

    ii = 0
    n_failed = 0

    while len(result_list) < 10_000:
        if n_failed > 5:
            print("[ERROR] Too many failed requests. Exiting.")
            break

        print(f"[INFO] Sending request {ii}")
        try:
            response = requests.request("GET", url, headers=headers, params=params)
            response_json = response.json()
            result_list = result_list + response_json["data"]
            params["next_token"] = response_json["meta"]["next_token"]
            print(
                f"[INFO] Received {len(result_list)} tweets so far.\n\t--> Head: {response_json['data'][0]['text']}"
            )
        except Exception as e:
            n_failed += 1
            print(
                f"[ERROR] in this request. Skipping it (missing {params.get('max_results')} tweets)"
            )
            print(response.status_code)
            print(response.json())
            print(e)
        finally:
            ii += 1
            time.sleep(0.2)

        if ii % 10 == 0:
            joblib.dump(result_list, f"resultlist_{hashtag}.joblib")

    joblib.dump(result_list, f"resultlist_{hashtag}.joblib")
    print("[INFO] Dumped final list.")

    result_df = pd.DataFrame(result_list)
    result_df.to_parquet(f"data/api_{hashtag}.parquet")
    print("[INFO] Saved final parquet.")

    return result_df


if __name__ == "__main__":
    # scrape_tweets_from_hashtags(hashtags)

    for hashtag in hashtags:
        print(f"{hashtag:=^80}")
        pull_tweets_for_hashtag(hashtag)
        print(f"[INFO] Done with {hashtag}. Sleeping for 60 seconds...")
        time.sleep(60)


#%%
