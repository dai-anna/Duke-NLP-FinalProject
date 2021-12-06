#%%
import pandas as pd
import joblib
from torchnlp.encoders.text import WhitespaceEncoder

with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)



