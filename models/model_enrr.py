from math import sqrt
import torch.nn as nn
import torch.nn.functional
from pytorch3d.ops import knn_points, knn_gather, ball_query, sample_farthest_points
from timm.models.layers import DropPath
from .utils import TransitionDown, MLP, PositionEncoder
import torch.nn.functional as F
from mamba_ssm import Mamba
import math

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
        self.norm = norm_layer(spatial_embeding_size) if norm_layer else None

    def forward(self, xytp):
        Fsp = self.spatial_embedding(xytp)
        Fte = self.temporal_embedding(xytp)
        F = torch.cat([Fsp, Fte], -1)
        F = self.proj(F)
        if self.norm:
            F = self.norm(F)
        return F
    
    
class LXformer(nn.Module):
    def __init__(self, in_channel, out_channel, k_l=16, height=260, width=346):
        super(LXformer, self).__init__()
        self.k_l = k_l
        self.height = height
        self.attn_scale = math.sqrt(out_channel)
        self.local_position_encoding = PositionEncoder(in_channel=4, out_channel=out_channel)
        self.local_transformations = nn.Linear(in_channel, out_channel * 3)
        self.local_norm = nn.LayerNorm(out_channel)
        self.softmax = nn.Softmax(dim=2)

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
    def __init__(self, height, width, in_channel, out_channel, k_l, filter_size):
        super(TransformerLayers, self).__init__()
        self.LXformer = LXformer(in_channel=in_channel, out_channel=out_channel, k_l=k_l)
        self.mamba = Mamba(d_model=out_channel, d_state=64, d_conv=4, expand=2)

    def forward(self, xytp, features):
        F_l = self.LXformer(xytp, features)
        F = self.mamba(F_l)
        return F


class TransformerLayerBlock(nn.Module):
    def __init__(self, h, w, dim, attn_dim, mlp_ratio, k_nearest,  conv_kernel_size, drop_path):
        super(TransformerLayerBlock, self).__init__()
        self.dim = dim
        mlp_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.transformerlayers = TransformerLayers(height=h, width=w, in_channel=dim, out_channel=attn_dim, k_l=k_nearest, filter_size=conv_kernel_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_channel=dim, hidden_channel=mlp_dim, out_channel=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, xytp, features):
        shortcut = features
        F_attn = self.transformerlayers(xytp, self.norm1(features))
        F_attn = F_attn + shortcut

        shortcut = F_attn
        F_attn = self.mlp(self.norm2(F_attn))
        F_attn = self.drop_path(F_attn) + shortcut
        return F_attn


class BasicLayer(nn.Module):
    def __init__(self, depth, down_stride, dim, attn_dim, mlp_ratio, k_nearest, h, w, conv_kernel_size, drop_path):
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerLayerBlock(
                h=h,
                w=w,
                dim=dim,
                attn_dim=attn_dim,
                mlp_ratio=mlp_ratio,
                k_nearest=k_nearest,
                conv_kernel_size=conv_kernel_size,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth))
        self.down = TransitionDown(in_ch=dim, out_ch=dim * 2, k_nearest=k_nearest,
                                   stride=down_stride) if down_stride > 0 else None

    def forward(self, xyzp, features):
        for blk in self.blocks:
            features = blk(xyzp, features)
        if self.down:
            xyzp, features = self.down(xyzp, features)
        return xyzp, features
    

class EventScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2, dropout=0.5):
        super(EventScorer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.mlp = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.mlp.append(nn.Linear(in_dim, hidden_dim))
        self.attention_fc = nn.Linear(hidden_dim, 1)
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, num_events, _ = x.size()
        for layer in self.mlp:
            x = F.relu(layer(x))
            x = self.dropout(x)
        attention_weights = F.softmax(self.attention_fc(x), dim=1) 
        x = x * attention_weights
        x = x.sum(dim=1)
        output = self.output_fc(x)
        output = torch.sigmoid(output)
        return output


class ENRR(nn.Module):
    def __init__(self,
                 embed_dim=32,
                 embed_norm=True,
                 num_classes=1,
                 drop_path_rate=0.2,
                 height=260,
                 width=346,
                 conv_ks_list=[5, 3],
                 depth_list=[1, 1],
                 k_nearest_list=[16, 16],
                 mlp_ratio_list=[4, 4],
                 down_stride_list=[2, -1]
                 ):
        super().__init__()
        self.embed = self.spatiotemporal_embedding = SpatiotemporalEmbedding(
            height=height, width=width, spatial_embeding_size=embed_dim, temporal_embedding_size=embed_dim, out_channel=32, norm_layer=nn.LayerNorm if embed_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth_list))]  

        self.enc = nn.ModuleList()
        for i in range(len(depth_list)):
            layer = BasicLayer(
                depth=depth_list[i],
                down_stride=down_stride_list[i],
                dim=int(embed_dim * 2 ** i),
                attn_dim=int(embed_dim * 2 ** i),
                mlp_ratio=mlp_ratio_list[i],
                k_nearest=k_nearest_list[i],
                h=height // (2 ** i),
                w=width // (2 ** i),
                conv_kernel_size=conv_ks_list[i],
                drop_path=dpr[sum(depth_list[:i]):sum(depth_list[:i + 1])])
            self.enc.append(layer)
            
        self.score = EventScorer(input_dim=64, hidden_dim=64, num_layers=2, dropout=0.5)
            
    def encoder(self, xyzp):
        f = self.embed(xyzp)
        for i, layer in enumerate(self.enc):
            xyzp, f = layer(xyzp, f)
        return f

    def forward(self, xyzp):
        f = self.encoder(xyzp)
        score = self.score(f)
        return score