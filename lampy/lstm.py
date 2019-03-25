import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        input, _ = self.lstm(input.view(-1, 1, 1).float())
        input = self.linear(input)
        return input


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import data

    lstm = LSTMModel(input_size=1, output_size=1)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(lstm.parameters())

    data_file = 'data/636f6a2e-3a48-4801-afce-05c864adc60e-lampy_2019.csv'
    sid = 'y7e4onsytkb3ydonflz5kcbcigkh5ulo'
    data_df = data.read_2018_data(data_file).loc[sid, 'pm10']
    dataset = data.LampyDataset(data_df)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for epoch in range(100):
        losses = [] 
        for seq in dataloader:
            lstm.zero_grad()
            prediction = lstm(seq['input'])
            loss = mse_loss(seq['target'], prediction.double())
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('epoch:\t{} MSE loss:\t{}'.format(epoch + 1, np.mean(losses)))
