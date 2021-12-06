import twint
import datetime
import os
import pandas as pd
import re


today = datetime.date.today()
# today = "2021-11-15"  # Hardcode


ROOT_DIR = "./"

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
        c.Output = f"twint_out_{trend}.csv"
        twint.run.Search(c)
