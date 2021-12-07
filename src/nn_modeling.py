#%%
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
batch_size = 32
num_epochs = 10

#%%
# Pad my input sequence with zeros
xtrain = nn.utils.rnn.pad_sequence(sequences = xtrain, batch_first=False, padding_value=0.0)
xval = nn.utils.rnn.pad_sequence(sequences = xval, batch_first=False, padding_value=0.0)
xtest = nn.utils.rnn.pad_sequence(sequences = xtest, batch_first=False, padding_value=0.0)

#%%
# Create data loader
class dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = nn.functional.one_hot(torch.Tensor(y.cat.codes.values).long())

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
# Define the GRU neural network
gru_model = nn.Sequential(
    nn.Embedding(num_embeddings=encoder.vocab_size, embedding_dim=embedding_dim),
    nn.GRU(
        input_size=embedding_dim,
        hidden_size=hidden_size,
        batch_first=True,
        bidirectional=True,
    ),
    nn.Linear(in_features=hidden_size * 2, out_features=len(hashtags)),
    nn.Softmax(),
)

#%%
# Check our model parameters
for i in gru_model.parameters():
    print(i)

# %%
optimizer = torch.optim.Adam(gru_model.parameters())
loss_function = nn.CrossEntropyLoss()
gru_model.train()
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):
        y_pred = gru_model(x)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch} loss: {loss.item()}")
    gru_model.eval()
    y_pred_val = gru_model(xval)
    loss_val = loss_function(y_pred_val, yval)
    print(f"Epoch {epoch} val loss: {loss_val.item()}")
    gru_model.train()
