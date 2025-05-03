from torch import nn


class Projection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.projection_layer = nn.Linear(args.patch_len, args.d_model)

    def forward(self, x):
        x = self.projection_layer(x)
        return x
