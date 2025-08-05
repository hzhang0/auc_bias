from torch import nn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

class MLP(nn.Module):
    def __init__(self, n_inputs, mlp_width, mlp_depth, mlp_dropout, n_outputs = 1):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, mlp_width)
        self.dropout = nn.Dropout(mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(mlp_width, mlp_width)
            for _ in range(mlp_depth-2)])
        self.output = nn.Linear(mlp_width, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
    
class MLPWithWeights(BaseEstimator):
    def __init__(self, mlp_width, mlp_depth, mlp_dropout,
                 batch_size, lr, max_epochs, device = 'cpu', debug = False):        
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.debug = debug
        self.max_epochs = max_epochs
        self.mlp_width = mlp_width
        self.mlp_depth = mlp_depth
        self.mlp_dropout = mlp_dropout
        
    def fit(self, X, y, sample_weight):
        n_inputs = X.shape[1]
        self.mlp = MLP(n_inputs, self.mlp_width, self.mlp_depth, self.mlp_dropout).to(self.device)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if sample_weight is None:
            train_ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1), torch.ones((len(y), 1)))
        else:
            train_ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(-1), torch.tensor(sample_weight).unsqueeze(-1))

        train_loader = DataLoader(train_ds, batch_size = self.batch_size, shuffle = True)
        optimizer = torch.optim.Adam(
            self.mlp.parameters(),
            lr=self.lr
        )     

        for epoch in range(self.max_epochs):
            self.mlp.train()
            train_loss = []
            for x, lab, weights in train_loader:
                x = x.float().to(self.device)
                lab = lab.float().to(self.device)
                pred = self.mlp(x)
                loss = F.binary_cross_entropy_with_logits(pred, lab, weight = weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            if self.debug:
                print(f'Epoch {epoch}; Train loss: {np.mean(train_loss)}')
        
        return self

    def predict(self, X):
        self.mlp.eval()
        return self.predict_proba(X)[:, 1].round()

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        X = X.float().to(self.device)
        self.mlp.eval()

        with torch.no_grad():
            out = torch.sigmoid(self.mlp(X)).squeeze(1).detach().cpu().numpy()
        return np.stack((1 - out, out), axis = 1)