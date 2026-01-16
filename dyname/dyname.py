
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from pdb import set_trace as st
import numpy as np
class TemperatureScaler(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaler, self).__init__()
        self.temperature = temperature

    def forward(self, logits):
        return logits / self.temperature

def clip(x, max_norm=1):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.clip(max_norm / x_norm, max=1)
    return x * scale, x_norm


class ExpertGate(nn.Module):
    def __init__(self, args, feature_dim):
        super().__init__()
        self.args = args
        self.feature_dim = feature_dim

        if self.args.weight_type == 0: ## 2-layer MLP
            self.final_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim//2),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(self.feature_dim//2, self.args.period_num+1),
                TemperatureScaler(temperature=self.args.krr_temperature),
                nn.Softmax(dim=-1)
            )
        elif self.args.weight_type == 1: ## weight as parameter
            self.weights = nn.Parameter(torch.randn(self.args.dec_in, self.args.period_num+1) * 0.01, requires_grad=True)

        elif self.args.weight_type == 2: ## 1-layer MLP
            self.final_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.args.period_num+1),
                TemperatureScaler(temperature=self.args.krr_temperature),
                nn.Softmax(dim=-1)
            )
        elif self.args.weight_type == 3: ## 1-layer MLP with direct input
            self.final_head = nn.Sequential(
                nn.Linear(self.args.seq_len, self.args.period_num+1),
                TemperatureScaler(temperature=self.args.krr_temperature),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        if self.args.clip_expert:
            x, x_norm = clip(x)
        return self.final_head(x)


class DynaME(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        if args.freeze:
            backbone.requires_grad_(False)

        self.lam = 0.95              
        self.delta = args.delta          
        self.scale = 1.0          
        self.signal_lam = 0.8
        self.eps = 1e-8
        self.ewma = None
        self.signal = 0.0

        self.backbone = backbone

        self.freeze_weight = True

        if self.args.model == 'PatchTST':
            self.feature_dim = self.backbone.model.head_nf
        elif self.args.model == 'iTransformer':
            self.feature_dim = self.args.d_model
        elif 'xPatch' in self.args.model:
            self.feature_dim = self.args.pred_len * 2


        self.expert_gate = ExpertGate(args, self.feature_dim)




    def _freeze_weights(self, flag):
        if flag:
            for param in self.expert_gate.parameters():
                param.requires_grad = False
            self.freeze_weight = True
        else:
            for param in self.expert_gate.parameters():
                param.requires_grad = True
            self.freeze_weight = False

        
    def _krr_solve(self, X, Y, rep, stats=None, explicit=False):
        """
        Solves Kernel Ridge Regression in its primal form.
        X: (num_samples, n_features, channel)
        Y: (num_samples, n_targets, channel)
        rep : (num_samples, n_features, channel)
        lmbda: Regularization parameter
        Returns W: (n_features, n_targets)
        """
        B, F, C = X.shape
        L = Y.shape[1]

        if stats is not None:
            means, stdev = stats
            Y = (Y - means) / stdev

        X = X.permute(2, 0, 1) # (C, B, F)
        Y = Y.permute(2, 0, 1) # (C, B, L)
        rep = rep.permute(2, 0, 1) # (C, 1, F)

        #K_train = torch.einsum('a f c, b f c -> c a b', X, X) # (C, B, B)
        #K_test = torch.einsum('a f c, b f c -> c a b', rep, X) # (C, 1, B)

        if explicit:
            XTX = torch.matmul(X.transpose(1, 2), X)
            XTY = torch.matmul(X.transpose(1, 2), Y)
            eye_F = torch.eye(F, device=X.device).unsqueeze(0)
            W = torch.linalg.solve(XTX + self.args.krr_lambda * eye_F, XTY)
            preds = torch.matmul(rep, W)

        else:
            K_train = torch.matmul(X, X.transpose(-1, -2))  # (C, B, B)
            K_test = torch.matmul(rep, X.transpose(-1, -2))   # (C, 1, B)

            trace_K = torch.diagonal(K_train, dim1=-2, dim2=-1).sum(-1)

            lmbda_scaled = self.args.krr_lambda * trace_K / B

            K_reg = K_train + lmbda_scaled[:, None, None] * torch.eye(B, device=K_train.device).unsqueeze(0)  # (C, B, B)


            alpha = torch.linalg.solve(K_reg, Y) # (C, B, L)
            preds = torch.matmul(K_test, alpha) # (C, 1, L)



        return preds.permute(1, 2, 0)
    
    def _get_period(self, x, fixed=False):
        if not fixed:
            assert x.shape[0] == 1
            length = x.shape[1]

            fft_result = torch.fft.rfft(x.permute(0, 2, 1), dim=-1)

            amplitude_spectrum = torch.abs(fft_result)

            integrated_spectrum = torch.mean(amplitude_spectrum, dim=1)

            top_k_amplitudes, top_k_indices = torch.topk(integrated_spectrum[:, 1:], k=self.args.period_num, dim=-1)

            if self.args.period_num == 1:
                period = [(length // (top_k_indices + 1)).squeeze().item()]
            else:
                period = sorted((length // (top_k_indices + 1)).squeeze().tolist(), reverse=True)


        else:
            period = [24, 168]

        return period[:self.args.period_num]

    def _get_rep(self, x):
        batch, channel = x.shape[0], x.shape[-1]
        if self.args.model in ['PatchTST', 'iTransformer', 'xPatch']:
            y, rep = self.backbone(x, return_emb=True)
            
        else:
            y, rep = x, x

        return y, rep.reshape(batch, -1, channel)
    
    def _revin(self, x):
        # Apply revin
        if self.args.revin:
            means = x.mean(1, keepdim=True).detach()
            x_norm = x - means
            stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = x_norm / stdev
            return x_norm, (means, stdev)
        else:
            return x, None


    def _update_ewma(self, mse):
        if self.ewma is None:
            self.ewma = mse
            self.signal = 0.0
            return self.signal

        self.ewma = (1 - self.lam) * mse + self.lam * self.ewma

        diff = mse - self.ewma

        new_signal = 1 - np.exp(-self.delta * (diff / self.scale)**2)
        
        self.signal = max(self.signal_lam * self.signal, new_signal)

        return self.signal
    
    def get_signal(self):
        return self.signal

    def get_ewma(self):
        return self.ewma

    def forward(self, batch_x, past_data=None, return_expert_preds=False):
    
        if self.freeze_weight:
            return self.backbone(batch_x)
        else:
            backbone_pred, rep = self._get_rep(batch_x)
            _, stats_test = self._revin(batch_x)

            if self.args.weight_type == 0 or self.args.weight_type == 2:
                expert_weights = self.expert_gate(rep.permute(0, 2, 1)) # (B, C, num_experts)
            elif self.args.weight_type == 1:
                expert_weights = F.softmax(self.expert_gate.weights.unsqueeze(0), dim=-1) # (1, C, num_experts)
            elif self.args.weight_type == 3:
                expert_weights = self.expert_gate(batch_x.permute(0, 2, 1)) # (B, C, num_experts)
            elif self.args.weight_type == 4: # simple average
                expert_weights = torch.ones((batch_x.shape[0], self.args.dec_in, self.args.period_num+1), device=batch_x.device) / (self.args.period_num+1)


            backbone_vector = torch.zeros_like(expert_weights)
            backbone_vector[:, :, 0] = 1.0
            #expert_weights = (1-self.args.beta) * expert_weights + self.args.beta * backbone_vector

            past_len = past_data.shape[1]

            expert_preds = []

            periods = self._get_period(batch_x.detach(), self.args.fixed_period)

            # Collect expert predictions for each period
        
            for j, period in enumerate(periods):
                if past_len - self.args.seq_len - period <= 0:
                    if return_expert_preds:
                        expert_preds.append(torch.zeros_like(batch_x[:, :self.args.pred_len, :]))
                    else:
                        expert_preds.append(torch.zeros_like(batch_x[:, :self.args.pred_len, :]))
                    continue

                indices = torch.arange(past_len - self.args.seq_len - period, 0, -period)
                indices = indices[(indices + self.args.seq_len + self.args.pred_len <= past_len)][:self.args.krr_train_num]

                if len(indices) == 0:
                    if return_expert_preds:
                        expert_preds.append(torch.zeros_like(batch_x[:, :self.args.pred_len, :]))
                    else:
                        expert_preds.append(torch.zeros_like(batch_x[:, :self.args.pred_len, :]))
                    continue

                lookback_x = []
                lookback_y = []
                for idx in indices:
                    lookback_x.append(past_data[:, idx:idx+self.args.seq_len, :])
                    lookback_y.append(past_data[:, idx+self.args.seq_len:idx+self.args.seq_len+self.args.pred_len, :])

                lookback_x = torch.cat(lookback_x, dim=0)
                lookback_y = torch.cat(lookback_y, dim=0)

                _, lookback_reps = self._get_rep(lookback_x)
                krr_preds = self._krr_solve(lookback_reps, lookback_y, rep, stats=stats_test)

                if stats_test is not None:
                    means, stdev = stats_test
                    krr_preds = krr_preds * stdev + means

                expert_preds.append(krr_preds)


            expert_preds = [backbone_pred] + expert_preds 

            expert_preds_tensor = torch.stack(expert_preds, dim=0).permute(1, 0, 2, 3)



            expert_preds_tensor_perm = expert_preds_tensor.permute(0, 1, 3, 2)  # (B, num_experts, C, pred_len)


            
            sig = self.get_signal()
            blending_factor_lambda = self.args.beta + sig * (1-self.args.beta)
            if self.args.weight_type != 4:
                expert_weights = (1-blending_factor_lambda) * expert_weights + blending_factor_lambda * backbone_vector



            # weighted sum over experts
            expert_weights_exp = expert_weights.permute(0, 2, 1).unsqueeze(3)  # (B, num_experts, C, 1)
            weighted_preds = (expert_preds_tensor_perm * expert_weights_exp).sum(dim=1)  # (B, C, pred_len)

            # permute back to (B, pred_len, C)
            total_pred = weighted_preds.permute(0, 2, 1)

            if self.args.plot_expert:
                return total_pred, backbone_pred, expert_weights, blending_factor_lambda, expert_preds
            else:
                return total_pred, backbone_pred, expert_weights, blending_factor_lambda
