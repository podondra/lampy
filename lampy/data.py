import pandas as pd
from torch.utils.data import Dataset


class LampyDataset(Dataset):
    def __init__(self, data_df):
        """The `data_df` is a pandas Series, where index is the
        `starttimestamp` as column is an attributte."""
        self.data_df = data_df
        # TODO preprocessing e.g. extracting features from timestamps

    def __len__(self):
        return self.data_df.shape[0] - 1

    def __getitem__(self, index):
        # TODO for now the label is the next measurement
        return {
                'input': self.data_df.iloc[index],
                'target': self.data_df.iloc[index + 1]
                }


def read_2018_data(data_file):
    return pd.read_csv(data_file, sep=';', parse_dates=['starttimestamp']) \
            .set_index(['sid', 'starttimestamp']).sort_index()


def read_2019_data(data_file):
    return pd.read_csv(data_file, parse_dates=['starttimestamp']) \
            .set_index(['sid', 'starttimestamp']).sort_index()


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_file = 'data/2e6bbd16-8d1c-4efa-b0f0-06a74f40d8d3-lampysensory-2019-1pololeti.csv'
    sid = 'y7e4onsytkb3ydonflz5kcbcigkh5ulo'
    data_df = read_2019_data(data_file).loc[sid, 'pm10']

    print(data_df.head())

    dataset = LampyDataset(data_df)

    print(len(dataset))
    print(dataset[16])

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in dataloader:
        print(batch)
        break
