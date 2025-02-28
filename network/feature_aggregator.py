import torch
import torch.nn as nn


def feature_aggretate(lh, rh, body, face):
    """
    Args:
        lh, rh, body, face: (batch_size, T, N, C) - Pose features per group
    Returns:
        (batch_size, T, 4C) - Aggregated feature representation
    """
    lh_pooled = lh.mean(dim=2).squeeze(-2)
    rh_pooled = rh.mean(dim=2).squeeze(-2)
    body_pooled = body.mean(dim=2).squeeze(-2)
    face_pooled = face.mean(dim=2).squeeze(-2)
    
    # Concatenate along feature dimension
    return torch.cat([lh_pooled, rh_pooled, body_pooled, face_pooled], dim=-1)  # (batch_size, T, 4C)


if __name__ == "__main__":
    batch_size = 4
    seq_length = 16
    feature_dim = 256

    num_keypoints = {"lh": 21, "rh": 21, "body": 9, "face": 18}

    # pose_agg = PoseFeatureAggregator()

    lh = torch.rand(batch_size, seq_length, num_keypoints["lh"], feature_dim)
    rh = torch.rand(batch_size, seq_length, num_keypoints["rh"], feature_dim)
    body = torch.rand(batch_size, seq_length, num_keypoints["body"], feature_dim)
    face = torch.rand(batch_size, seq_length, num_keypoints["face"], feature_dim)

    output = feature_aggretate(lh, rh, body, face)
    print(output.shape)