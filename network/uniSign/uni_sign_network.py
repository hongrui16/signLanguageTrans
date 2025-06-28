import torch
import torch.nn as nn
import numpy as np


from network.uniSign.pose_encoder import PoseEncoder
from network.uniSign.feature_aggregator import feature_aggretate
from network.uniSign.LLM_trans import SignLanguageLLM
from network.uniSign.temporal_encoder import STGCNTemporalEncoder

from utils.mediapipe_kpts_mapping import MediapipeKptsMapping


class UniSignNetwork(nn.Module):
    """Full Uni-Sign Model: Separate Encoders for Pose and Temporal Processing"""
    def __init__(self, kpts_dim = 2, hidden_dim=256, device = 'cpu',  **kwargs):
        super(UniSignNetwork, self).__init__()

        self.device = device
        freeze_llm = kwargs.get("freeze_llm", False)
        pose_set = kwargs.get("pose_set", 'hand_body')
        llm_name = kwargs.get("llm_name", "mbart-large-50")  # Default to MBart50
        self.llm_name = llm_name

        self.pose_set = pose_set
        
        hand_adj_matrix = MediapipeKptsMapping.hand_adj_matrix
        face_adj_matrix = MediapipeKptsMapping.face_adj_matrix
        body_adj_matrix = MediapipeKptsMapping.body_adj_matrix

        hand_adj_matrix = torch.tensor(hand_adj_matrix, dtype=torch.float32).to(self.device)
        face_adj_matrix = torch.tensor(face_adj_matrix, dtype=torch.float32).to(self.device)
        body_adj_matrix = torch.tensor(body_adj_matrix, dtype=torch.float32).to(self.device)

        # Separate Pose Encoders for different parts
        self.pose_encoder_lh = PoseEncoder(kpts_dim, hand_adj_matrix)
        self.pose_encoder_rh = PoseEncoder(kpts_dim, hand_adj_matrix)
        self.pose_encoder_body = PoseEncoder(kpts_dim, body_adj_matrix)
        
        # Separate Temporal Encoders (ST-GCN) for different parts
        self.temporal_encoder_lh = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=hand_adj_matrix)
        self.temporal_encoder_rh = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=hand_adj_matrix)
        self.temporal_encoder_body = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=body_adj_matrix)
        
        if 'face' in pose_set:
            # If face is included in the pose set, use the face encoder            
            self.pose_encoder_face = PoseEncoder(kpts_dim, face_adj_matrix)
            self.temporal_encoder_face = STGCNTemporalEncoder(in_features=256, out_dim=hidden_dim, adj=face_adj_matrix)


    
        self.llm_trans = SignLanguageLLM(llm_name, freeze_llm = freeze_llm)

    def forward(self, lh, rh, body, face = None, split = 'train', decoder_input_ids = None):
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

        # Temporal Encoding for each part
        lh_features = self.temporal_encoder_lh(lh_features)
        rh_features = self.temporal_encoder_rh(rh_features)
        body_features = self.temporal_encoder_body(body_features)
        
        if face is None or 'face' not in self.pose_set:
            sign_features = feature_aggretate(lh_features, rh_features, body_features)            
        else:
            # Feature Aggregation
            face_features = self.pose_encoder_face(face)
            face_features = self.temporal_encoder_face(face_features)
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
    kpts_dim = 3

    num_keypoints = {"lh": 21, "rh": 21, "body": 9, "face": 18}

    adj_lh = torch.eye(num_keypoints["lh"])
    adj_rh = torch.eye(num_keypoints["rh"])
    adj_body = torch.eye(num_keypoints["body"])
    adj_face = torch.eye(num_keypoints["face"])

    model = UniSignNetwork(num_keypoints)

    lh = torch.rand(batch_size, seq_length, num_keypoints["lh"], kpts_dim)
    rh = torch.rand(batch_size, seq_length, num_keypoints["rh"], kpts_dim)
    body = torch.rand(batch_size, seq_length, num_keypoints["body"], kpts_dim)
    face = torch.rand(batch_size, seq_length, num_keypoints["face"], kpts_dim)

    output = model(lh, rh, body, face, adj_lh, adj_rh, adj_body, adj_face)
    print("LLM Output Shape:", output.shape)
