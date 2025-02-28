import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_features, out_features, adj = None):
        super(GraphConvolution, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

        self.adj = adj  # adj: (N, N) - Adjacency matrix

    def forward(self, x):
        """
        Args:
            x: (batch_size, T, N, C) - Pose features
            adj: (N, N) - Adjacency matrix defining keypoint connectivity
        Returns:
            (batch_size, T, N, out_features)
        """
        adj = self.adj

        batch_size, T, N, C = x.shape  # Extract shape

        # Reshape to (batch_size * T, N, C) for matrix multiplication
        x = x.view(batch_size * T, N, C)  

        # Apply adjacency matrix multiplication across each frame separately
        x = torch.einsum('ij,bnc->bnc', adj, x)  # Efficient batch-wise operation
        # print(f'x.shape: {x.shape}') # (batch_size * T, N, C)
        # Apply linear transformation
        x = self.fc(x)  # (batch_size * T, N, out_features)

        # Reshape back to (batch_size, T, N, out_features)
        x = x.view(batch_size, T, N, -1)

        return F.relu(x)


class PoseEncoder(nn.Module):
    """Pose Encoder using a 3-layer Spatial GCN"""
    def __init__(self, feature_dim, adj = None):

        super(PoseEncoder, self).__init__()
        # Step 1: Initial Linear Projection (from raw keypoint 2D to 64-d)
        self.linear = nn.Linear(feature_dim, 64)

        # Step 2: Three-layer GCN (spatial modeling)
        self.gc1 = GraphConvolution(64, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.gc2 = GraphConvolution(64, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.gc3 = GraphConvolution(128, 256)
        self.bn3 = nn.BatchNorm2d(256)

        self.adj = adj # Adjacency matrix (N, N)

    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, T, N, C). 
               where T: sequence length, N: number of keypoints, C: feature dim (x, y). For 2D keypoints, C=2 (x, y)..
            adj: Adjacency matrix (N, N) defining keypoint connections.
        
        Returns:
            Tensor of shape (batch_size, T, N, 256)
        """
        adj = self.adj
        
        batch_size, T, N, C = x.shape  # Keep original shape

        # Step 1: Flatten for Linear Projection
        x = x.view(-1, C)  # (batch_size * T * N, C=2)

        # Step 2: Linear projection (C=2 → C=64)
        x = self.linear(x)  # (batch_size * T * N, 64)

        # Step 3: Reshape back to (batch_size, T, N, 64)
        x = x.view(batch_size, T, N, -1)

        # Step 4: Pass to GCN
        x = self.gc1(x, adj)  # (batch_size, T, N, 64)
        x = self.bn1(x)
        x = self.gc2(x, adj)  # (batch_size, T, N, 128)
        x = self.bn2(x)
        x = self.gc3(x, adj)  # (batch_size, T, N, 256)
        x = self.bn3(x)

        return x  # (batch_size, T, N, 256)


class GCN_encoder(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=2):
        super(GCN_encoder, self).__init__()
        
        # 第一层 GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二层 GCN
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # 第三层 GCN
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x: 节点特征矩阵 [21, in_channels]，这里 in_channels=2 (x, y 坐标)
        # edge_index: 图的连接关系 [2, num_edges]
        
        # 第一层
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第三层
        x = self.conv3(x, edge_index)
        
        return x

# 定义手的固定拓扑结构（edge_index）
def get_hand_edge_index():
    # 手的 21 个关键点之间的连接（基于常见手的骨架结构，例如 MediaPipe 的手部模型）
    # 关键点编号通常为：0（腕部），1-4（拇指），5-8（食指），9-12（中指），13-16（无名指），17-20（小指）
    edges = [
        # 腕部到各手指根部
        (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
        # 拇指
        (1, 2), (2, 3), (3, 4),
        # 食指
        (5, 6), (6, 7), (7, 8),
        # 中指
        (9, 10), (10, 11), (11, 12),
        # 无名指
        (13, 14), (14, 15), (15, 16),
        # 小指
        (17, 18), (18, 19), (19, 20)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index

# 示例使用
def main_1():
    batch_size = 4
    # 假设输入是 21 个 2D 关键点
    num_nodes = 21
    in_channels = 2  # 输入特征维度 (x, y 坐标)
    hidden_channels = 128  # 隐藏层维度
    out_channels = 256  # 输出维度 (例如预测调整后的 x, y 坐标)
    
    # 随机生成手的 2D 关键点数据
    x = torch.rand([batch_size, num_nodes, in_channels])
    
    # 获取手的拓扑结构
    edge_index = get_hand_edge_index()
    
    # 初始化模型
    model = GCN_encoder(in_channels, hidden_channels, out_channels)
    
    # 前向传播
    output = model(x, edge_index)
    print("输入形状:", x.shape)  # [21, 2]
    print("输出形状:", output.shape)  # [21, 2]
    
def main_2():
    batch_size = 4
    seq_length = 16
    num_keypoints = 21  # e.g., hands (21*2) + body (9) + face (18)
    feature_dim = 2  # (x, y) coordinates
    
    adj_matrix = np.zeros((21, 21), dtype=int)

    # 定义连接关系
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),   # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),   # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]

    # 填充邻接矩阵
    for (i, j) in edges:
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # 无向图，双向连接



    pose_encoder = PoseEncoder(num_keypoints, feature_dim)
    
    # Simulated input (random keypoint data)
    keypoint_data = torch.rand(batch_size, seq_length, num_keypoints, feature_dim)

    # Example adjacency matrix (fully connected for simplicity, can be customized)

    adj_matrix = torch.Tensor(adj_matrix)
    encoded_features = pose_encoder(keypoint_data, adj_matrix)
    print("Encoded feature shape:", encoded_features.shape)  # Expected: (batch_size, T, N, out_dim)

    
# Example Usage
if __name__ == "__main__":
    main_1()