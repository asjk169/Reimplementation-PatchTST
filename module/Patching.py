from torch import nn


class Patching(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.patch_len = args.patch_len
        self.stride = args.stride

    def forward(self, x):
        x = x.unfold(-1, self.patch_len, self.stride)
        return x
