from torch import nn
import torch


class Pos_embidding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_num = (args.seq_len - args.patch_len) // args.stride + 2
        # Create an empty matrix for position values
        pe = torch.zeros(self.patch_num, args.d_model)
        # Assign position values
        position = torch.arange(0, self.patch_num).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.d_model, 2) * -(torch.log(torch.tensor(10000.0)) / args.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer, so it will not be updated
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe.unsqueeze(0)
        return x
