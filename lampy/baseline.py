import torch.nn as nn


class BaselineModel(nn.Module):
    """Baseline model predict the next value to be the same as previous
    value."""
    def forward(self, x):
        # x is the previous value in a time serie and the predicted value
        # in the baseline model is the same
        return x


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    import data

    data_file = 'data/2e6bbd16-8d1c-4efa-b0f0-06a74f40d8d3-lampysensory-2019-1pololeti.csv'
    sid = 'y7e4onsytkb3ydonflz5kcbcigkh5ulo'
    data_df = data.read_2019_data(data_file).loc[sid, 'pm10']
    dataset = data.LampyDataset(data_df)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    baseline_model = BaselineModel()
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for data in dataloader:
            prediction = baseline_model(data['input'])
            print('MSE:', mse_loss(data['target'], prediction).item())

    plt.plot(data['target'].numpy(), label='target', marker='o')
    plt.plot(prediction.numpy(), label='prediction', marker='.')
    plt.legend()
    plt.grid()
    plt.show()
