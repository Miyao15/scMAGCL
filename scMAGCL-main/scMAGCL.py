import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
import copy
from torch_geometric.nn.conv import GCNConv
from utils import device


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, h, phi, dropout=0):
        super(GraphConstructor, self).__init__()

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


class MultiScaleGraphConstructor(nn.Module):
    def __init__(self, input_dim, h, phi, phi1=0.05, dropout=0):
        super(MultiScaleGraphConstructor, self).__init__()

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)
        self.phi1 = nn.Parameter(torch.tensor(phi1), requires_grad=False)

    def _attention_scores(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        d_k = query.size(-1)
        scores = torch.bmm(query.squeeze(2).permute(1, 0, 2), key.squeeze(2).permute(1, 2, 0)) / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn

    def forward(self, query, key):
        attns = self._attention_scores(query, key)

        phi_m = self.phi
        delta = torch.clamp(self.phi1, min=0.0)
        phi_c = torch.clamp(phi_m - delta, min=0.0)
        phi_f = torch.clamp(phi_m + delta, max=0.9999)

        adj_c = torch.where(attns >= phi_c, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))
        adj_m = torch.where(attns >= phi_m, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))
        adj_f = torch.where(attns >= phi_f, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj_c, adj_m, adj_f


def DataAug(x, adj, prob_feature, prob_edge):
    batch_size = x.shape[0]
    input_dim = x.shape[1]

    tensor_p = torch.ones((batch_size, input_dim)) * (1 - prob_feature)
    mask_feature = torch.bernoulli(tensor_p).to(device)

    tensor_p = torch.ones((batch_size, batch_size)) * (1 - prob_edge)
    mask_edge = torch.bernoulli(tensor_p).to(device)

    return mask_feature * x, mask_edge * adj


def multiscale_contrastive_loss(z1, z2, adj_c, adj_m, adj_f, adj_c_aug, adj_m_aug, adj_f_aug, 
                               tau=0.8, alpha=0.55, beta=0.4):
    """
    多尺度对比损失：利用不同尺度的互补信息
    """
    # 1. 基于不同尺度的对比损失
    loss_coarse = final_cl_loss(alpha, beta, z1, z2, adj_c, adj_c_aug, tau, hidden_norm=True)
    loss_medium = final_cl_loss(alpha, beta, z1, z2, adj_m, adj_m_aug, tau, hidden_norm=True)
    loss_fine = final_cl_loss(alpha, beta, z1, z2, adj_f, adj_f_aug, tau, hidden_norm=True)
    
    # 2. 跨尺度对比损失（粗-细对比，增强多尺度一致性）
    loss_cross_cf = final_cl_loss(alpha, beta, z1, z2, adj_c, adj_f_aug, tau * 0.8, hidden_norm=True)
    loss_cross_mf = final_cl_loss(alpha, beta, z1, z2, adj_m, adj_f_aug, tau * 0.9, hidden_norm=True)
    
    # 3. 自适应权重组合
    # 权重基于各尺度的信息量
    coarse_info = adj_c.sum()
    medium_info = adj_m.sum()
    fine_info = adj_f.sum()
    total_info = coarse_info + medium_info + fine_info + 1e-8
    
    w_c = coarse_info / total_info
    w_m = medium_info / total_info
    w_f = fine_info / total_info
    
    # 4. 组合损失
    scale_loss = w_c * loss_coarse + w_m * loss_medium + w_f * loss_fine
    cross_loss = 0.2 * (loss_cross_cf + loss_cross_mf)  # 跨尺度损失权重较小
    
    return scale_loss + cross_loss


def byol_loss(z1, z2, hidden_norm=True):
    """
    BYOL损失：Bootstrap Your Own Latent
    不需要负样本，通过预测目标网络的表示来学习
    """
    if hidden_norm:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
    
    # BYOL损失：最小化预测与目标之间的均方误差
    # 对称化：z1预测z2 + z2预测z1
    loss_12 = 2 - 2 * (z1 * z2).sum(dim=1).mean()
    loss_21 = 2 - 2 * (z2 * z1).sum(dim=1).mean()
    
    return (loss_12 + loss_21) / 2


def momentum_update(online_net, target_net, tau=0.99):
    """
    动量更新目标网络
    """
    for online_param, target_param in zip(online_net.parameters(), target_net.parameters()):
        target_param.data = tau * target_param.data + (1 - tau) * online_param.data


class MLPProjector(nn.Module):
    """
    BYOL的MLP投影头和预测头
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class BYOLPredictor(nn.Module):
    """
    BYOL的预测头
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BYOLPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def sim(z1, z2, hidden_norm):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def cl_loss(z, z_aug, adj, tau, hidden_norm=True):
    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    positive = inter_view_sim.diag() + (intra_view_sim.mul(adj)).sum(1) + (inter_view_sim.mul(adj)).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) * 2 + 1
    loss = torch.log(loss) / adj_count

    return -torch.mean(loss, 0)


def final_cl_loss(alpha1, alpha2, z, z_aug, adj, adj_aug, tau, hidden_norm=True):
    loss = alpha1 * cl_loss(z, z_aug, adj, tau, hidden_norm) + alpha2 * cl_loss(z_aug, z, adj_aug, tau, hidden_norm)

    return loss


class Model(nn.Module):
    def __init__(self, input_dim, graph_head, phi, gcn_dim, mlp_dim,
                 prob_feature, prob_edge, tau, alpha, beta, dropout,
                 phi1=0.05,
                 use_byol: bool = True, byol_hidden_dim: int = 512,
                 byol_output_dim: int = 256, momentum_tau: float = 0.996):
        super(Model, self).__init__()
        self.prob_feature = prob_feature
        self.prob_edge = prob_edge
        self.tau = tau
        self.alpha = alpha
        self.beta = beta

        self.use_byol = use_byol
        self.momentum_tau = momentum_tau

        self.ms_graphconstructor = MultiScaleGraphConstructor(input_dim, graph_head, phi, phi1=phi1, dropout=0)
        self.gcn = GCNConv(input_dim, gcn_dim)
        self.w_imp = nn.Linear(gcn_dim, input_dim)
        self.mlp = nn.Linear(gcn_dim, mlp_dim)

        self.dropout = nn.Dropout(p=dropout)

        if self.use_byol:
            self.online_projector = MLPProjector(mlp_dim, byol_hidden_dim, byol_output_dim)
            self.online_predictor = BYOLPredictor(byol_output_dim, byol_hidden_dim, byol_output_dim)
            self.target_projector = MLPProjector(mlp_dim, byol_hidden_dim, byol_output_dim)
            self.target_projector.load_state_dict(self.online_projector.state_dict())
            for param in self.target_projector.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.dropout(x)
        adj_c, adj_m, adj_f = self.ms_graphconstructor(x, x)
        adj = torch.clamp(adj_c + adj_m + adj_f, max=1)
        adj = adj - torch.diag_embed(adj.diag())
        edge_index = torch.nonzero(adj == 1).T

        x_aug, adj_aug = DataAug(x, adj, self.prob_feature, self.prob_edge)
        edge_index_aug = torch.nonzero(adj_aug == 1).T

        z = self.gcn(x, edge_index)
        z_aug = self.gcn(x_aug, edge_index_aug)

        x_imp = self.w_imp(z)
        z_mlp = self.mlp(z)
        z_mlp_aug = self.mlp(z_aug)

        mask_edge = (adj_aug > 0).float()
        adj_c_aug = adj_c * mask_edge
        adj_m_aug = adj_m * mask_edge
        adj_f_aug = adj_f * mask_edge

        loss_cl = multiscale_contrastive_loss(
            z_mlp, z_mlp_aug,
            adj_c, adj_m, adj_f,
            adj_c_aug, adj_m_aug, adj_f_aug,
            tau=self.tau, alpha=self.alpha, beta=self.beta
        )
        
        if self.use_byol:
            online_proj_orig = self.online_projector(z_mlp)
            online_pred_orig = self.online_predictor(online_proj_orig)
            online_proj_aug = self.online_projector(z_mlp_aug)
            online_pred_aug = self.online_predictor(online_proj_aug)
            with torch.no_grad():
                target_proj_orig = self.target_projector(z_mlp)
                target_proj_aug = self.target_projector(z_mlp_aug)
            loss_byol = (
                byol_loss(online_pred_orig, target_proj_aug, hidden_norm=True) + 
                byol_loss(online_pred_aug, target_proj_orig, hidden_norm=True)
            ) / 2
        else:
            loss_byol = torch.tensor(0.0, device=z_mlp.device)

        return z, x_imp, loss_cl, loss_byol

    def update_target_network(self):
        if self.use_byol:
            momentum_update(self.online_projector, self.target_projector, self.momentum_tau)
