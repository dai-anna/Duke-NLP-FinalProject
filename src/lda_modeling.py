#%%
import pandas as pd
import joblib
from torchnlp.encoder import WhitespaceEncoder

with open("../artefacts/encoder.pickle", "rb") as f:
    encoder: WhitespaceEncoder = joblib.load(f)



