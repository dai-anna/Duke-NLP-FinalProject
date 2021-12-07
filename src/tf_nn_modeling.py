#%%
import tensorflow as tf
from tensorflow.python.keras.layers.wrappers import Bidirectional
import torch
import torch.nn as nn
import joblib
from preprocessing_helpers import *
from data_collecting import hashtags

#%%
# Load data from disk
encoder = joblib.load("../artefacts/encoder.pickle")

train = pd.read_parquet("../data/train.parquet")
val = pd.read_parquet("../data/val.parquet")
test = pd.read_parquet("../data/test.parquet")

xtrain, ytrain = encode_dataframe(encoder, data=train, mode="pytorch")
xval, yval = encode_dataframe(encoder, data=val, mode="pytorch")
xtest, ytest = encode_dataframe(encoder, data=test, mode="pytorch")

#%%
# Specify hyperparameters
embedding_dim = 32
hidden_size = 64
batch_size = 512
num_epochs = 10
hidden_dense_dim = 32
#%%
# Pad my input sequence with zeros
xtrain = nn.utils.rnn.pad_sequence(sequences=xtrain, batch_first=True, padding_value=0.0)
xval = nn.utils.rnn.pad_sequence(sequences=xval, batch_first=True, padding_value=0.0)
xtest = nn.utils.rnn.pad_sequence(sequences=xtest, batch_first=True, padding_value=0.0)

#%%
train_dataset = tf.data.Dataset.from_tensor_slices(
    (xtrain, ytrain.cat.codes.values)
).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval.cat.codes.values)).batch(
    batch_size
)


#%%
from tensorflow import keras

model = keras.Sequential(
    [
        keras.layers.Embedding(input_dim=encoder.vocab_size, output_dim=embedding_dim),
        keras.layers.Bidirectional(keras.layers.GRU(hidden_size, return_sequences=False)),
        keras.layers.Dense(units=hidden_dense_dim, activation="relu"),
        keras.layers.Dense(7, activation="softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)

hist = model.fit(train_dataset, validation_data=val_dataset, epochs=2)

#%%
pd.DataFrame(hist.history)[["loss", "val_loss"]].plot(figsize=(8, 5))
