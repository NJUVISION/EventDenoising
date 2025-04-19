import torch.nn as nn
import torch
from pytorch3d.ops import knn_points, knn_gather, sample_farthest_points

class MLP(nn.Module):
    def __init__(self, in_channel, hidden_channel=None, out_channel=None, act_layer=nn.GELU):
        super(MLP, self).__init__()
        out_channel = out_channel or in_channel
        hidden_channel = hidden_channel or in_channel
        self.net = nn.Sequential(
            nn.Linear(in_channel, hidden_channel),
            act_layer(),
            nn.Linear(hidden_channel, out_channel)
        )

    def forward(self, x):
        return self.net(x)


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
    
class SwapAxes(nn.Module):
    def __init__(self, dim1: int = 1, dim2: int = 2):
        self.dim1 = dim1
        self.dim2 = dim2
        super().__init__()

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class TransitionDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_nearest: int, stride: int):
        super().__init__()
        self.stride = stride
        self.k_nearest = k_nearest
        self.mlp = nn.Sequential(
            nn.Linear(4 + in_ch, out_ch),
            SwapAxes(1, 3),
            nn.BatchNorm2d(out_ch),
            SwapAxes(1, 3),
            nn.ReLU()
        )

    def forward(self, xyzp, features):
        xyz = xyzp[:, :, :3]
        new_xyz, _ = sample_farthest_points(xyz, K=xyzp.shape[1] // self.stride)
        knn_result = knn_points(new_xyz, xyz, K=self.k_nearest)
        idx = knn_result.idx
        new_xyzp = knn_gather(xyzp, idx)[:, :, 0]

        grouped_xyzp = knn_gather(xyzp, idx)
        grouped_xyzp_norm = grouped_xyzp - new_xyzp[:, :, None]

        grouped_features = knn_gather(features, idx) 
        new_features_concat = torch.cat([grouped_xyzp_norm, grouped_features], dim=-1) 
        new_features_concat = self.mlp(new_features_concat)  
        new_features_concat = torch.max(new_features_concat, 2)[0]  
        return new_xyzp, new_features_concat