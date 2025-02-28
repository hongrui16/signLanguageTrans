import torch
import torch.nn as nn
import numpy as np


from network.pose_encoder import PoseEncoder
from network.feature_aggregator import feature_aggretate
from network.LLM_trans import SignLanguageLLM
from network.temporal_encoder import STGCNTemporalEncoder

hand_adj_matrix = np.zeros((21, 21), dtype=int)

# 定义连接关系
'''
hand keypoints indices (0–20)
0: Wrist
1: Thumb CMC
2: Thumb MCP
3: Thumb IP
4: Thumb Tip
5: Index Finger MCP
6: Index Finger PIP
7: Index Finger DIP
8: Index Finger Tip
9: Middle Finger MCP
10: Middle Finger PIP
11: Middle Finger DIP
12: Middle Finger Tip
13: Ring Finger MCP
14: Ring Finger PIP
15: Ring Finger DIP
16: Ring Finger Tip
17: Little Finger MCP
18: Little Finger PIP
19: Little Finger DIP
20: Little Finger Tip

'''
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),   # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
]

# 填充邻接矩阵
for (i, j) in edges:
    hand_adj_matrix[i, j] = 1
    hand_adj_matrix[j, i] = 1  # 无向图，双向连接


'''
face Keypoint Indices (0–17)
Face outline (0–5):
0: Left temple
1: Right temple
2: Left cheek
3: Right cheek
4: Chin (bottom)
5: Nose bridge (top of face outline)
Mouth (6–17):
6: Left mouth corner
7: Right mouth corner
8–11: Upper lip (4 points, e.g., top center, left, right, middle)
12–15: Lower lip (4 points, e.g., bottom center, left, right, middle)
16–17: Additional mouth points (e.g., inner lip edges)

'''

# Define the number of keypoints (18 for face)
n_keypoints_faces = 18

# Initialize a zero matrix (18x18) on the appropriate device (e.g., CPU or GPU)
face_adj_matrix = np.zeros((n_keypoints_faces, n_keypoints_faces), dtype=torch.float32)  # Use torch.float32 for consistency

# Define connections (edges) for 18 facial keypoints
edges = [
    # Face outline
    (0, 2), (2, 4), (4, 3), (3, 1),  # Jawline
    (0, 5), (1, 5),  # Nose bridge to temples
    
    # Mouth
    (6, 7),  # Mouth corners
    (6, 8), (8, 10), (10, 7),  # Upper lip outer
    (6, 12), (12, 14), (14, 7),  # Lower lip outer
    (8, 9), (9, 10),  # Upper lip inner
    (12, 13), (13, 14),  # Lower lip inner
    (16, 8), (16, 10),  # Inner upper lip
    (17, 12), (17, 14)  # Inner lower lip
]

# Fill the adjacency matrix (undirected graph, so symmetric)
for (i, j) in edges:
    face_adj_matrix[i, j] = 1
    face_adj_matrix[j, i] = 1  # Ensure symmetry for undirected graph

'''
body Keypoint Indices (0–8)
0: Left wrist
1: Left elbow
2: Left shoulder
3: Right wrist
4: Right elbow
5: Right shoulder
6: Left ear
7: Right ear
8: Face center point (midpoint between ears or nose bridge, representing the face)

'''
n_keypoints_body = 9

# Initialize a zero matrix (9x9) on the appropriate device (e.g., CPU or GPU)
body_adj_matrix = torch.zeros((n_keypoints_body, n_keypoints_body), dtype=torch.float32)  # Use torch.float32 for consistency

# Define connections (edges) for 9 body keypoints
edges = [
    # Left arm
    (0, 1), (1, 2),  # Left wrist → left elbow → left shoulder
    # Right arm
    (3, 4), (4, 5),  # Right wrist → right elbow → right shoulder
    # Head/Face
    (6, 8), (7, 8),  # Left ear → face center, right ear → face center
    # Upper body (optional)
    (2, 6), (5, 7)   # Left shoulder → left ear, right shoulder → right ear
]

# Fill the adjacency matrix (undirected graph, so symmetric)
for (i, j) in edges:
    body_adj_matrix[i, j] = 1
    body_adj_matrix[j, i] = 1  # Ensure symmetry for undirected graph



class UniSignNetwork(nn.Module):
    """Full Uni-Sign Model: Separate Encoders for Pose and Temporal Processing"""
    def __init__(self, num_keypoints, hidden_dim=256, LLM_name="facebook/mbart-large-50", device = 'cpu',  **kwargs):
        super(UniSignNetwork, self).__init__()

        self.device = device
        tokenizer = kwargs.get("tokenizer", None)
        hand_adj_matrix = torch.tensor(hand_adj_matrix, dtype=torch.float32).to(self.device)
        face_adj_matrix = torch.tensor(face_adj_matrix, dtype=torch.float32).to(self.device)
        body_adj_matrix = torch.tensor(body_adj_matrix, dtype=torch.float32).to(self.device)

        # Separate Pose Encoders for different parts
        self.pose_encoder_lh = PoseEncoder(num_keypoints["lh"], hand_adj_matrix)
        self.pose_encoder_rh = PoseEncoder(num_keypoints["rh"], hand_adj_matrix)
        self.pose_encoder_body = PoseEncoder(num_keypoints["body"], body_adj_matrix)
        self.pose_encoder_face = PoseEncoder(num_keypoints["face"], face_adj_matrix)

        # Separate Temporal Encoders (ST-GCN) for different parts
        self.temporal_encoder_lh = STGCNTemporalEncoder(num_keypoints["lh"], in_features=256, out_dim=hidden_dim, adj=hand_adj_matrix)
        self.temporal_encoder_rh = STGCNTemporalEncoder(num_keypoints["rh"], in_features=256, out_dim=hidden_dim, adj=hand_adj_matrix)
        self.temporal_encoder_body = STGCNTemporalEncoder(num_keypoints["body"], in_features=256, out_dim=hidden_dim, adj=body_adj_matrix)
        self.temporal_encoder_face = STGCNTemporalEncoder(num_keypoints["face"], in_features=256, out_dim=hidden_dim, adj=face_adj_matrix)

        self.llm_trans = SignLanguageLLM(LLM_name, tokenizer = tokenizer)

    def forward(self, lh, rh, body, face, split = 'train', decoder_input_ids = None):
        """
        Args:
            lh, rh, body, face: (batch_size, T, N, C) - Pose inputs
        Returns:
            LLM Output
        """

        # Pose Encoding for each part
        lh_features = self.pose_encoder_lh(lh)
        rh_features = self.pose_encoder_rh(rh)
        body_features = self.pose_encoder_body(body)
        face_features = self.pose_encoder_face(face)

        # Temporal Encoding for each part
        lh_features = self.temporal_encoder_lh(lh_features)
        rh_features = self.temporal_encoder_rh(rh_features)
        body_features = self.temporal_encoder_body(body_features)
        face_features = self.temporal_encoder_face(face_features)

        # Feature Aggregation
        sign_features = feature_aggretate(lh_features, rh_features, body_features, face_features)

        # LLM Translation
        return self.llm_trans(sign_features, model = split, decoder_input_ids = decoder_input_ids)  # LLM output embeddings


# from sacrebleu import corpus_bleu

# # 测试阶段
# translated_text, encoder_hidden = model(sign_features, mode="test")

# # 假设 reference_texts 是参考翻译列表（每个样本可能有多个参考）
# reference_texts = [["This is a test sentence"], ["Another test sentence"]]  # 示例
# bleu_score = corpus_bleu(translated_text, reference_texts)
# print("BLEU score:", bleu_score.score)


if __name__ == "__main__":
    batch_size = 4
    seq_length = 16
    feature_dim = 3

    num_keypoints = {"lh": 21, "rh": 21, "body": 9, "face": 18}

    adj_lh = torch.eye(num_keypoints["lh"])
    adj_rh = torch.eye(num_keypoints["rh"])
    adj_body = torch.eye(num_keypoints["body"])
    adj_face = torch.eye(num_keypoints["face"])

    model = UniSignNetwork(num_keypoints)

    lh = torch.rand(batch_size, seq_length, num_keypoints["lh"], feature_dim)
    rh = torch.rand(batch_size, seq_length, num_keypoints["rh"], feature_dim)
    body = torch.rand(batch_size, seq_length, num_keypoints["body"], feature_dim)
    face = torch.rand(batch_size, seq_length, num_keypoints["face"], feature_dim)

    output = model(lh, rh, body, face, adj_lh, adj_rh, adj_body, adj_face)
    print("LLM Output Shape:", output.shape)
