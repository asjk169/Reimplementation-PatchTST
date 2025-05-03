from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from torch.utils.data import DataLoader


class Dataset(Dataset):
    def __init__(self, flag, args):
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        df = pd.read_csv(args.file_path).drop(columns='date')
        type_map = {'train': 0, 'vali': 1, 'test': 2}
        self.set_type = type_map[flag]
        if args.data in ['ETTm1', 'ETTm2']:
            border_adj = 4
        else:
            border_adj = 1
        border1s = [0, 12 * 30 * 24 * border_adj - self.seq_len, 12 * 30 * 24 * border_adj + 4 * 30 * 24 * border_adj - self.seq_len]
        border2s = [12 * 30 * 24 * border_adj, 12 * 30 * 24 * border_adj + 4 * 30 * 24 * border_adj, 12 * 30 * 24 * border_adj + 8 * 30 * 24 * border_adj]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        train = df[border1s[0]: border2s[0]]
        scaler = StandardScaler()
        scaler.fit(train)
        df = scaler.transform(df)
        df = df[border1:border2]
        self.df = torch.tensor(df, dtype=torch.float32)

    def __len__(self):
        return len(self.df) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        seq_begin = index
        seq_end = seq_begin + self.seq_len
        pred_begin = seq_end
        pred_end = pred_begin + self.pred_len
        seq_x = self.df[seq_begin:seq_end]
        seq_y = self.df[pred_begin:pred_end]
        return seq_x, seq_y


def get_data(flag, args):
    dataset = Dataset(flag, args)
    if flag == 'test':
        shuffle = False
    else:
        shuffle = True
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=shuffle, num_workers=args.n_workers, drop_last=True)
    print(f'{flag}: {len(dataset)}')
    return loader
