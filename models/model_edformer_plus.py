import torch.nn as nn
import torch
from pytorch3d.ops import knn_points, knn_gather, ball_query
from .utils import MLP
import math
from mamba_ssm import Mamba

TIMESTAMP_COLUMN = 0
X_COLUMN = 1
Y_COLUMN = 2
POLARITY_COLUMN = 3


class SpatialEmbedding(nn.Module):
    def __init__(self, height, width, K, out_channel):
        super(SpatialEmbedding, self).__init__()
        self.embedding = MLP(in_channel=K*2, hidden_channel=out_channel*2, out_channel=out_channel)
        self.height = height
        self.width = width
        self.K = K
    
    def _get_self_idx(self, k_idx):
        b, n, k = k_idx.shape
        if not hasattr(self, 'idx') or self.idx.shape != k_idx.shape or self.idx.device != k_idx.device:
            self.idx = torch.arange(n, device=k_idx.device)[
                None, :, None].repeat(b, 1, k)
        return self.idx

    def forward(self, xytp):
        xyt = xytp[..., :3].clone().detach()
        xyt[..., 0] = 0
        temp = ball_query(xyt, xyt, radius=5/self.height, K=self.K)
        idx = temp.idx
        idx = torch.where(idx == -1, self._get_self_idx(idx), idx)
        xy = xytp[..., [1, 2]]
        delta = xy[:, :, None] - knn_gather(xy, idx)
        B, N, _, _ = delta.size()
        embedding_tensor = delta.view(B, N, -1)
        Fsp = self.embedding(embedding_tensor)
        return Fsp


class TemporalEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TemporalEmbedding, self).__init__()
        self.pos_embed = nn.Linear(in_channel, out_channel)
        self.neg_embed = nn.Linear(in_channel, out_channel)

    def forward(self, xytp):
        xytp = xytp.clone().detach()
        pos_neg = xytp[..., POLARITY_COLUMN][..., None]
        timestamp = xytp[..., TIMESTAMP_COLUMN][..., None]
        Fte = self.pos_embed(timestamp) * pos_neg + self.neg_embed(timestamp) * (1 - pos_neg)
        return Fte


class SpatiotemporalEmbedding(nn.Module):
    def __init__(self, height, width, spatial_embeding_size, temporal_embedding_size, out_channel, norm_layer=None):
        super(SpatiotemporalEmbedding, self).__init__()
        self.spatial_embedding = SpatialEmbedding(
            height=height, width=width, K=9, out_channel=spatial_embeding_size)
        self.temporal_embedding = TemporalEmbedding(in_channel=1, out_channel=temporal_embedding_size)
        self.proj = nn.Linear(spatial_embeding_size + temporal_embedding_size, out_channel, bias=False)
        self.norm = nn.LayerNorm(out_channel)

    def forward(self, xytp):
        Fsp = self.spatial_embedding(xytp)
        Fte = self.temporal_embedding(xytp)
        F = torch.cat([Fsp, Fte], -1)
        F = self.proj(F)
        if self.norm:
            F = self.norm(F)
        return F


class PositionEncoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PositionEncoder, self).__init__()
        self.linear1 = nn.Linear(in_channel, in_channel)
        self.batch_norm = nn.BatchNorm1d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        x = self.linear1(x)
        b, n, k, f = x.shape
        x = self.batch_norm(x.reshape([b * n, k, f]).transpose(1, 2)).transpose(1, 2)
        x = self.linear2(self.relu(x)).reshape([b, n, k, -1])
        return x


class LXformer(nn.Module):
    def __init__(self, in_channel, out_channel, k_l=16, height=260, width=346):
        super(LXformer, self).__init__()
        self.k_l = k_l
        self.height = height
        self.attn_scale = math.sqrt(out_channel)
        self.local_position_encoding = PositionEncoder(in_channel=4, out_channel=32)
        self.local_transformations = nn.Linear(in_channel, out_channel * 3)
        self.local_norm = nn.LayerNorm(in_channel)
        self.softmax = nn.Softmax(dim=2)
        
    def _get_self_idx(self, k_idx):
        b, n, k = k_idx.shape
        if not hasattr(self, 'idx') or self.idx.shape != k_idx.shape or self.idx.device != k_idx.device:
            self.idx = torch.arange(n, device=k_idx.device)[
                None, :, None].repeat(b, 1, k)
        return self.idx

    def forward(self, xytp, features):
        xyt = xytp[:, :, :3].clone().detach()
        idx = knn_points(xyt, xyt, K=self.k_l).idx
        delta = self.local_position_encoding(xytp[:, :, None] - knn_gather(xytp, idx))
        local_transformations = self.local_transformations(features)
        c = local_transformations.shape[-1] // 3
        varphi, psi, alpha = local_transformations[...,
                                                   :c], local_transformations[..., c:2 * c], local_transformations[..., 2 * c:]
        psi, alpha = knn_gather(psi, idx), knn_gather(alpha, idx)
        local_attn = self.softmax(self.local_norm(
            varphi[:, :, None, :] - psi + delta) / self.attn_scale) * (alpha + delta)
        local_attn = torch.sum(local_attn, dim=2)
        return local_attn


class TransformerLayers(nn.Module):
    def __init__(self, height, width, in_channel, out_channel, k_l):
        super(TransformerLayers, self).__init__()
        self.LXformer = LXformer(in_channel=in_channel, out_channel=out_channel, k_l=k_l)
        self.mamba = Mamba(d_model=out_channel, d_state=64, d_conv=4, expand=2)

    def forward(self, xytp, features):
        attn_l = self.LXformer(xytp, features)
        attn = self.mamba(attn_l)
        return attn


class TransformerLayerBlock(nn.Module):
    def __init__(self, height, width, in_channel, out_channel, k_l):
        super(TransformerLayerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.transformerlayers = TransformerLayers(
            height=height, width=width, in_channel=in_channel, out_channel=out_channel, k_l=k_l)
        self.mlp = MLP(in_channel=in_channel, hidden_channel=in_channel*4, out_channel=out_channel)
        self.drop_path = nn.Identity()

    def forward(self, xytp, features):
        shortcut = features
        F_attn = self.transformerlayers(xytp, self.norm1(features))
        F_attn = F_attn + shortcut

        shortcut = F_attn
        F_attn = self.mlp(self.norm2(F_attn))
        F_attn = self.drop_path(F_attn) + shortcut
        return F_attn


class EDformerPlus(nn.Module):
    def __init__(self, height=260, width=346):
        super(EDformerPlus, self).__init__()
        self.spatial_embeding_size = 32
        self.temporal_embedding_size = 32
        self.spatiotemporal_embedding = SpatiotemporalEmbedding(
            height=height, width=width, spatial_embeding_size=self.spatial_embeding_size, temporal_embedding_size=self.temporal_embedding_size, out_channel=32, norm_layer=True)
        self.transformer_layers_Large = TransformerLayerBlock(
            height=height, width=width, in_channel=32, out_channel=32, k_l=16)
        self.head = nn.Linear(32, 1)

    def forward(self, xytp):
        B, N, _ = xytp.size()
        F_L = self.spatiotemporal_embedding(xytp)
        F = self.transformer_layers_Large(xytp, F_L)
        prob = self.head(F)
        return prob
