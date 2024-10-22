import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


class LSTM(nn.Module):
    """Implementation of LSTM in PyTorch."""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """Initialize an LSTM with input of dimension `input_size`, hidden
        vector of dimension `hidden_size`, output of dimension `output_size`
        and `num_layers` of hidden layers."""
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # initialize LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # initialize linear model that takes hidden vector
        # and transforms it to output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Pass forward."""
        lstm_out, hidden = self.lstm(x.unsqueeze(0), hidden)
        out = self.fc(lstm_out)
        return out.squeeze(0), hidden

    def to_tensor(self, array):
        """Convert numpy array to PyTorch tensor."""
        return torch.from_numpy(array.astype(np.float32))

    def predict(self, X):
        """Predict for each value in sequence `X`."""
        tensor = self.to_tensor(X)
        # turn off gradients
        with torch.no_grad():
            y_pred, _ = self.forward(tensor, hidden=None)
        return y_pred.numpy()

    def train(self, X, y, X_val, y_val, n_epochs, seq_len, verbose=True):
        """Train this LSTM with `X` and `y` with validation data `X_val`
        and `y_val` for `n_epochs` epochs on sequences of length `seq_len`.
        Loss function is MSE loss and optimize is Adam"""
        # create dataset and loader of data
        dataset = TensorDataset(self.to_tensor(X), self.to_tensor(y))
        loader = DataLoader(dataset, batch_size=seq_len, shuffle=False)
        X_val, y_val = self.to_tensor(X_val), self.to_tensor(y_val)

        # loss function is MSE loss
        criterion = nn.MSELoss(reduction='sum')
        # optimizer is Adam with default parameter setting
        optimizer = optim.Adam(self.parameters())

        # list for keeping training history
        tr_losses, val_losses = [], = []
        for epoch in range(1, n_epochs + 1):
            hidden = None
            tr_loss = 0
            for seq_X, seq_y in loader:
                seq_y_pred, hidden = self.forward(seq_X, hidden)
                loss = criterion(seq_y_pred, seq_y)
                tr_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                # TODO maybe clip gradients
                optimizer.step()

                # disconnect due to gradient
                hidden = tuple([h.data for h in hidden])

            with torch.no_grad():
                pred_val, _ = self.forward(X_val, hidden=None)
                val_loss = criterion(pred_val, y_val).item()

            tr_losses.append(tr_loss / X.shape[0])
            val_losses.append(val_loss / X_val.shape[0])
            if verbose:
                print('epoch: {} train loss: {} validation loss: {}'.format(
                      epoch, tr_losses[-1], val_losses[-1]))

        return {'tr_losses': tr_losses, 'val_losses': val_losses}
