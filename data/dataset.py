import torch
import pandas as pd
from torch.utils.data import Dataset

class tttDataset(Dataset):
    def __init__(self, path: str, len_rep: int):
        self.X_data = torch.tensor(pd.read_csv(path).to_numpy()[:, :len_rep]).float()
        self.y_data = torch.tensor(pd.read_csv(path).to_numpy()[:, len_rep:]).long().squeeze(1)
        self.num_datapoints = self.X_data.shape[0]

    def __len__(self):
        return self.y_data.shape[0]

    def __getitem__(self, idx: int):
        return self.X_data[idx], self.y_data[idx]

class binDataset(Dataset):
    def __init__(self, num: int, length: int):
        ''' 
        synthetic dataset of two coloring of a length-dimensional hypercube
        
        length is the number of binary elements in the dataset

        dataset is 2^length elements long

        num ranges from 0 to 2^(2^length), acting as an index for the coloring
        
        '''
        self.num = num
        self.length = length
        # create dataset
        y_bin = [int(i) for i in format(num, f'0{2**self.length}b')]

        self.X_data = []
        self.y_data = []
        # 
        for i in range(2**self.length):
            bin_str = format(i, f'0{length}b')
            self.X_data.append([int(i) for i in bin_str])
            self.y_data.append(y_bin[i])

        self.X_data = torch.tensor(self.X_data, dtype=torch.float32)
        self.y_data = torch.tensor(self.y_data, dtype=torch.long)

    def __len__(self):
        return 2**self.length

    def __getitem__(self, idx: int):
        return self.X_data[idx], self.y_data[idx]

if __name__ == '__main__':
    num = 31
    length = 5
    dataset = binDataset(num=num, length=length)
    print(f'Dataset for num={num}, length={length}')
    print(dataset.X_data)
    print(dataset.X_data.shape)
    print(dataset.y_data)
    print(dataset.y_data.shape)
