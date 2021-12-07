#%%
import pandas as pd
import torch
import torch.nn as nn
import joblib
from preprocessing_helpers import encode_dataframe
from data_collecting import hashtags

#%%
# Load data from disk
encoder = joblib.load("../artefacts/encoder.pickle")

train = pd.read_parquet("../data/train.parquet")
val = pd.read_parquet("../data/val.parquet")
test = pd.read_parquet("../data/test.parquet")

train = train.assign(lenoftweet=train.tweet.apply(lambda x: len(x.split())))
train = train.query("lenoftweet <= 60")

xtrain, ytrain = encode_dataframe(encoder, data=train, mode="pytorch")
xval, yval = encode_dataframe(encoder, data=val, mode="pytorch")
xtest, ytest = encode_dataframe(encoder, data=test, mode="pytorch")

#%%
# Specify hyperparameters
embedding_dim = 32
hidden_size = 64
batch_size = 1024
num_epochs = 500
hidden_dense_dim = 32

#%%
# Pad my input sequence with zeros
xtrain = nn.utils.rnn.pad_sequence(
    sequences=xtrain, batch_first=True, padding_value=0.0
)
xval = nn.utils.rnn.pad_sequence(sequences=xval, batch_first=True, padding_value=0.0)
xtest = nn.utils.rnn.pad_sequence(sequences=xtest, batch_first=True, padding_value=0.0)

#%%
# Create data loader
class dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x.long()
        self.y = nn.functional.one_hot(torch.Tensor(y.cat.codes.values).long()).float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


train_dataset = dataset(xtrain, ytrain)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

val_dataset = dataset(xval, yval)

#%%
# Define the nn models
gru_model = nn.Sequential(
    nn.Embedding(num_embeddings=encoder.vocab_size, embedding_dim=embedding_dim),
    nn.GRU(
        input_size=embedding_dim,
        hidden_size=hidden_size,
        batch_first=True,
        bidirectional=True,
    ),
)
softmax_model = nn.Sequential(
    nn.Linear(in_features=hidden_size * 2, out_features=hidden_dense_dim),
    nn.ReLU(),
    nn.Linear(in_features=hidden_dense_dim, out_features=len(hashtags)),
    nn.Softmax(),
)

# %%
# Train the model
optimizer = torch.optim.Adam(
    list(gru_model.parameters()) + list(softmax_model.parameters())
)
loss_function = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    gru_model.train()
    current_epoch_losses = []
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()

        y_pred = gru_model(x)[1]

        y_pred = torch.cat((y_pred[0], y_pred[1]), dim=-1)
        y_pred = softmax_model(y_pred)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        current_epoch_losses.append(loss.item())
    gru_model.eval()

    y_pred_val = gru_model(val_dataset.x)[1]
    # y_pred_val = y_pred_val.view(val_dataset.x.shape[0], -1)
    y_pred_val = torch.cat((y_pred_val[0], y_pred_val[1]), dim=-1)
    y_pred_val = softmax_model(y_pred_val)
    loss_val = loss_function(y_pred_val, val_dataset.y)
    print(
        f"Epoch {epoch} loss: {sum(current_epoch_losses)/len(current_epoch_losses)}, val loss: {loss_val.item()}"
    )

# %%
