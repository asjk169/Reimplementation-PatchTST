import torch.nn as nn


class Flatten_linear_head(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_len = args.pred_len
        self.patch_num = (args.seq_len - args.patch_len) // args.stride + 2
        self.linear_head = nn.Linear(self.patch_num * args.d_model, args.pred_len)
        self.dropout = nn.Dropout(args.fc_dropout)

    def forward(self, x):   # batch, channel_in, d_model, patch_num
        x = x.flatten(-2)   # batch, channel_in, (d_model * patch_num)
        x = self.linear_head(x)     # batch, channel_in, pred_len
        return x
