#%%
import torch
import torch.nn as nn
from torchnlp.encoders.text import WhitespaceEncoder
import joblib

#%%
encoder = joblib.load("../artefacts/encoder.pickle")
X = 

embedding_dim = 50
hidden_size = 500

#%%
# pad my inputs with zeros
nn.utils.rnn.pad_sequence(sequences = X_validation, batch_first=False, padding_value=0.0)
#%%

model = nn.Sequential(
    nn.Embedding(num_embeddings=encoder.vocab_size, embedding_dim=embedding_dim),
)