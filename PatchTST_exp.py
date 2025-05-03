from model.PatchTST_model import PatchTST_model
from dataloader.dataloader import get_data
import torch.nn as nn
import torch
import numpy as np
import random
import torch.nn.functional as F
import time
from copy import deepcopy


class PatchTST_exp():
    def __init__(self, args):
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.args = args
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PatchTST_model(args).to(self.device)

    def train(self):
        train_loader = get_data('train', self.args)
        vali_loader = get_data('vali', self.args)
        test_loader = get_data('test', self.args)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        train_step = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        self.args.lr,
                                                        epochs=self.args.epochs,
                                                        steps_per_epoch=train_step,
                                                        pct_start=self.args.pct_start)
        best_mse = float('inf')
        early_stop_count = 0
        epochs = self.args.epochs
        best_state_dict = deepcopy(self.model.state_dict())
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = []
            self.model.train()
            for seq, label in train_loader:
                optimizer.zero_grad()
                seq, label = seq.to(self.device), label.to(self.device)
                pred = self.model(seq)      # batch, pred_len, channel_in
                loss = criterion(pred, label)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
            self.model.eval()
            with torch.no_grad():
                vali_loss = []
                for seq, label in vali_loader:
                    seq, label = seq.to(self.device), label.to(self.device)
                    pred = self.model(seq)
                    loss = criterion(pred, label)
                    vali_loss.append(loss.item())
            avg_train_loss = np.average(train_loss)
            avg_vali_loss = np.average(vali_loss)
            if avg_vali_loss < best_mse:
                best_mse = avg_vali_loss
                best_state_dict = deepcopy(self.model.state_dict())
                early_stop_count = 0
            else:
                early_stop_count += 1
            end_time = time.time()
            print(f'Epoch: {epoch + 1}, train_loss: {avg_train_loss:.3f}, vali_loss: {avg_vali_loss:.3f}, best_score: {best_mse:.3f}, time_spend: {end_time - start_time}')
            print(f"Updating learning rate to {optimizer.param_groups[0]['lr']}")
            if early_stop_count >= 18:
                print(f'early stopped with best_mse:{best_mse:.3f}')
                break
        self.model.load_state_dict(best_state_dict)
        with torch.no_grad():
            test_loss = []
            test_mae = []
            for seq, label in test_loader:
                seq, label = seq.to(self.device), label.to(self.device)
                pred = self.model(seq)
                loss = criterion(pred, label)
                test_loss.append(loss.item())
                mae = F.l1_loss(pred, label, reduction='mean')
                test_mae.append(mae.item())
            avg_test_loss = np.average(test_loss)
            avg_test_mae = np.average(test_mae)
            print(f'test_mse:{avg_test_loss:.3f}, test_mae: {avg_test_mae:.3f}')
        return avg_test_loss, avg_test_mae
