import torch.nn as nn
import torch.nn.functional as F
import torch


class Transformer_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.head_dim = args.d_model // args.n_heads
        self.Q = nn.Linear(args.d_model, args.d_model)
        self.K = nn.Linear(args.d_model, args.d_model)
        self.V = nn.Linear(args.d_model, args.d_model)
        self.scale = nn.Parameter(torch.tensor(
            self.head_dim ** -0.5), requires_grad=False)
        self.attention_linear = nn.Linear(args.d_model, args.d_model)
        self.dropout1 = nn.Dropout(args.dropout)
        self.norm1 = nn.BatchNorm1d(args.d_model)
        self.ff = nn.Sequential(nn.Linear(args.d_model, args.d_ff),
                                nn.GELU(),
                                nn.Linear(args.d_ff, args.d_model))
        self.dropout2 = nn.Dropout(args.dropout)
        self.norm2 = nn.BatchNorm1d(args.d_model)

    def forward(self, x):
        q = self.Q(x).view(self.args.batch * self.args.channel_in, self.args.patch_num, self.args.n_heads, self.head_dim).transpose(-2, -3)
        k = self.K(x).view(self.args.batch * self.args.channel_in, self.args.patch_num, self.args.n_heads, self.head_dim).transpose(-2, -3)
        v = self.V(x).view(self.args.batch * self.args.channel_in, self.args.patch_num, self.args.n_heads, self.head_dim).transpose(-2, -3)
        attention_scores = q @ k.transpose(-2, -1) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_results = attention_weights @ v
        attention_results = attention_results.transpose(-2, -3).contiguous().view(self.args.batch * self.args.channel_in, self.args.patch_num, self.args.d_model)
        attention_results = self.attention_linear(attention_results)
        attention_results = x + self.dropout1(attention_results)  # batch * channel_in, patch_num, d_model
        attention_results = attention_results.transpose(-1, -2)   # batch * channel_in, d_model, patch_num
        attention_results = self.norm1(attention_results)         # batch * channel_in, d_model, patch_num
        attention_results = attention_results.transpose(-1, -2)   # batch * channel_in, patch_num, d_model
        encoder_output = self.ff(attention_results)     # batch * channel_in, patch_num, d_model
        encoder_output = attention_results + self.dropout2(encoder_output)      # batch * channel_in, patch_num, d_model
        encoder_output = encoder_output.transpose(-1, -2)       # batch * channel_in, d_model, patch_num
        encoder_output = self.norm2(encoder_output)     # batch * channel_in, d_model, patch_num
        encoder_output = encoder_output.transpose(-1, -2)        # batch * channel_in, patch_num, d_model
        return encoder_output
