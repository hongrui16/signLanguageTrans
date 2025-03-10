import torch
import torch.nn as nn
import numpy as np


from network.pose_encoder import PoseEncoder
from network.feature_aggregator import feature_aggretate
from network.LLM_trans import SignLanguageLLM
from network.temporal_encoder import STGCNTemporalEncoder

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping


class UniSignNetwork(nn.Module):
    """Full Uni-Sign Model: Separate Encoders for Pose and Temporal Processing"""
    def __init__(self, feature_dim = 2, hidden_dim=256, LLM_name="facebook/mbart-large-50", device = 'cpu',  **kwargs):
        super(UniSignNetwork, self).__init__()

        self.device = device
        tokenizer = kwargs.get("tokenizer", None)
        hand_adj_matrix = MediapipeKptsMapping.hand_adj_matrix
        face_adj_matrix = MediapipeKptsMapping.face_adj_matrix
        body_adj_matrix = MediapipeKptsMapping.body_adj_matrix

        hand_adj_matrix = torch.tensor(hand_adj_matrix, dtype=torch.float32).to(self.device)
        face_adj_matrix = torch.tensor(face_adj_matrix, dtype=torch.float32).to(self.device)
        body_adj_matrix = torch.tensor(body_adj_matrix, dtype=torch.float32).to(self.device)

        # Separate Pose Encoders for different parts
        self.pose_encoder_lh = PoseEncoder(feature_dim, hand_adj_matrix)
        self.pose_encoder_rh = PoseEncoder(feature_dim, hand_adj_matrix)
        self.pose_encoder_body = PoseEncoder(feature_dim, body_adj_matrix)
        self.pose_encoder_face = PoseEncoder(feature_dim, face_adj_matrix)

        # Separate Temporal Encoders (ST-GCN) for different parts
        self.temporal_encoder_lh = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=hand_adj_matrix)
        self.temporal_encoder_rh = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=hand_adj_matrix)
        self.temporal_encoder_body = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=body_adj_matrix)
        self.temporal_encoder_face = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=face_adj_matrix)

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
        return self.llm_trans(sign_features, mode = split, decoder_input_ids = decoder_input_ids)  # LLM output embeddings


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
