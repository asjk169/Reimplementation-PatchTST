from torch import nn
from module import RevIN, Patching, Projection, Pos_embidding, Transformer_encoder, Flatten_linear_head


class PatchTST_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.patch_num = (args.seq_len - args.patch_len) // args.stride + 2
        self.args = args
        self.instance_norm = RevIN(args)
        self.padding_patch_layer = nn.ReplicationPad1d((0, args.stride))
        self.patching = Patching(args)
        self.projection = Projection(args)
        self.pos_embidding = Pos_embidding(args)
        self.dropout1 = nn.Dropout(args.dropout)
        self.transformer_encoder = nn.ModuleList(
            [Transformer_encoder(args) for _ in range(args.n_layers)]
        )
        self.flatten_linear_head = Flatten_linear_head(args)

    def forward(self, x):   # batch, seq_len, channel_in
        x = self.instance_norm(x, 'norm')   # batch, seq_len, channel_in
        x = x.transpose(-1, -2)  # batch, channel_in, seq_len
        x = self.padding_patch_layer(x)  # batch, channel_in, (seq_len + stride)
        x = self.patching(x)    # batch, channel_in, patch_num, patch_len
        x = self.projection(x)  # batch, channel_in, patch_num, d_model
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (batch * channel_in), patch_num, d_model
        x = self.pos_embidding(x)   # (batch * channel_in), patch_num, d_model
        x = self.dropout1(x)    # (batch * channel_in), patch_num, d_model
        for layer in self.transformer_encoder:
            x = layer(x)    # (batch * channel_in), patch_num, d_model
        x = x.reshape(-1, self.args.channel_in, x.shape[-2], x.shape[-1])  # batch, channel_in, patch_num, d_model
        x = x.transpose(-1, -2)  # batch, channel_in, d_model, patch_num
        x = self.flatten_linear_head(x)     # batch, channel_in, pred_len
        x = x.transpose(-1, -2)      # batch, pred_len, channel_in
        x = self.instance_norm(x, 'denorm')     # batch, pred_len, channel_in
        return x
