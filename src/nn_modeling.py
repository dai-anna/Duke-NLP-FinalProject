#%%
# Import packages
import pandas as pd
import torch
import torch.nn as nn
import joblib
from preprocessing_helpers import encode_dataframe
from data_collecting import hashtags

import os
torch.set_num_threads(16)

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
batch_size = 64
num_epochs = 10
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

class combined(nn.Module):
    def __init__(self):
        super(combined, self).__init__()
        self.gru_model = gru_model
        self.softmax_model = softmax_model

    def forward(self, x):
        x = self.gru_model(x)[1]
        x = torch.cat((x[0], x[1]), dim=-1)
        x = self.softmax_model(x)
        return x

# %%
# Train the model
optimizer = torch.optim.Adam(
    list(gru_model.parameters()) + list(softmax_model.parameters()), lr=0.001
)
loss_function = nn.CrossEntropyLoss()

# #%%
# from torch_lr_finder import LRFinder
# lr_finder = LRFinder(combined(), optimizer, loss_function, device="cpu")
# lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
# lr_finder.plot() # to inspect the loss-learning rate graph
# lr_finder.reset() # to reset the model and optimizer to their initial state

#%%
performance_metrics = {
    "train_loss": [],
    "val_loss": [],
}
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
    y_pred_val = torch.cat((y_pred_val[0], y_pred_val[1]), dim=-1)
    y_pred_val = softmax_model(y_pred_val)
    loss_val = loss_function(y_pred_val, val_dataset.y)

    avg_trainloss_over_batches = sum(current_epoch_losses)/len(current_epoch_losses)
    print(
        f"Epoch {epoch} loss: {avg_trainloss_over_batches}, val loss: {loss_val.item()}"
    )
    performance_metrics["train_loss"].append(avg_trainloss_over_batches)
    performance_metrics["val_loss"].append(loss_val.item())

# %%
pd.DataFrame(performance_metrics).plot()
# %%
