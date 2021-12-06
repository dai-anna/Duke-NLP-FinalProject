import twint

hashtags = [
    "crypto",
    "tesla",
    "gsw",
    "formula1",
    "thanksgiving",
    "holidays",
    "covid19",
]

def scrape_tweets_from_hashtags(hashtags):

    for _, trend in enumerate(hashtags):
        c = twint.Config()
        c.Search = "#" + trend  # your search here
        c.Lang = "en"
        c.Limit = 10_000
        c.Store_csv = True
        c.Output = f"data/twint_output_{trend}.csv"
        twint.run.Search(c)

if __name__ == "__main__":
    scrape_tweets_from_hashtags(hashtags)