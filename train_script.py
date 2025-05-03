from PatchTST_exp import PatchTST_exp
import argparse
import sys
import torch
import pandas as pd


# PatchTST/42
df = pd.DataFrame(columns=['Dataset', 'Pred_len', 'MSE', 'MAE'])


# ETTh1
for pred_len in (96, 192, 336, 720):
    args = argparse.Namespace(
        seed=2021,
        data='ETTh1',
        file_path='dataset/ETTh1.csv',
        batch=128,
        channel_in=7,
        seq_len=336,
        pred_len=pred_len,
        patch_len=16,
        stride=8,
        d_model=16,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.3,
        fc_dropout=0.3,
        n_workers=10,
        epochs=100,
        lr=0.0001,
        pct_start=0.3,
    )
    log_file = 'log/PactchTST_' + args.data + '_' + str(args.seq_len) + '_' + str(args.pred_len) + '.log'
    with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
        sys.stdout = f
        exp = PatchTST_exp(args)
        test_mse, test_mae = exp.train()
        sys.stdout = sys.__stdout__
    torch.cuda.empty_cache()
    df.loc[len(df)] = [args.data, pred_len, test_mse, test_mae]

# ETTh2
for pred_len in (96, 192, 336, 720):
    args = argparse.Namespace(
        seed=2021,
        data='ETTh2',
        file_path='dataset/ETTh2.csv',
        batch=128,
        channel_in=7,
        seq_len=336,
        pred_len=pred_len,
        patch_len=16,
        stride=8,
        d_model=16,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        dropout=0.3,
        fc_dropout=0.3,
        n_workers=10,
        epochs=100,
        lr=0.0001,
        pct_start=0.3,
    )
    log_file = 'log/PactchTST_' + args.data + '_' + str(args.seq_len) + '_' + str(args.pred_len) + '.log'
    with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
        sys.stdout = f
        exp = PatchTST_exp(args)
        test_mse, test_mae = exp.train()
        sys.stdout = sys.__stdout__
    torch.cuda.empty_cache()
    df.loc[len(df)] = [args.data, pred_len, test_mse, test_mae]

# ETTm1
for pred_len in (96, 192, 336, 720):
    args = argparse.Namespace(
        seed=2021,
        data='ETTm1',
        file_path='dataset/ETTm1.csv',
        batch=128,
        channel_in=7,
        seq_len=336,
        pred_len=pred_len,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
        fc_dropout=0.2,
        n_workers=10,
        epochs=100,
        lr=0.0001,
        pct_start=0.4,
    )
    log_file = 'log/PactchTST_' + args.data + '_' + str(args.seq_len) + '_' + str(args.pred_len) + '.log'
    with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
        sys.stdout = f
        exp = PatchTST_exp(args)
        test_mse, test_mae = exp.train()
        sys.stdout = sys.__stdout__
    torch.cuda.empty_cache()
    df.loc[len(df)] = [args.data, pred_len, test_mse, test_mae]

# ETTm2
for pred_len in (96, 192, 336, 720):
    args = argparse.Namespace(
        seed=2021,
        data='ETTm2',
        file_path='dataset/ETTm2.csv',
        batch=128,
        channel_in=7,
        seq_len=336,
        pred_len=pred_len,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=16,
        n_layers=3,
        d_ff=256,
        dropout=0.2,
        fc_dropout=0.2,
        n_workers=10,
        epochs=100,
        lr=0.0001,
        pct_start=0.4,
        lr_adj='Auto'
    )
    log_file = 'log/PactchTST_' + args.data + '_' + str(args.seq_len) + '_' + str(args.pred_len) + '.log'
    with open(log_file, 'w', buffering=1, encoding='utf-8') as f:
        sys.stdout = f
        exp = PatchTST_exp(args)
        test_mse, test_mae = exp.train()
        sys.stdout = sys.__stdout__
    torch.cuda.empty_cache()
    df.loc[len(df)] = [args.data, pred_len, test_mse, test_mae]

df.to_csv('results.csv', index=False)
