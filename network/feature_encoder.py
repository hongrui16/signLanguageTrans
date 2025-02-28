import os
import sys
import torch
import torch.nn as nn

class MLPLayers(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, num_layers=2, device='cpu'):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU()) # inplace = False 会报错
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.mlp.to(device)

    def forward(self, x):
        return self.mlp(x)

class DINOv2FeatureEncoder(nn.Module):
    def __init__(self, encoder_name='dinov2_vits14', use_patch_tokens=True, device='cpu'):
        """
        DINOv2 Feature Encoder for 3D Pose Estimation
        
        Args:
            encoder_name (str): The DINOv2 model name ('dinov2_vits14' by default)
            out_dim (int): The output feature dimension (default=768 for cls + patch avg)
            use_patch_tokens (bool): Whether to include patch tokens (default=True)
        """
        super().__init__()

        # Load DINOv2 model
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', encoder_name)
        self.dinov2.to(device)

        # Feature projection (384 -> out_dim)
        input_dim = 384 if not use_patch_tokens else 768  # 384 for cls, 768 for cls + avg_patch

        self.use_patch_tokens = use_patch_tokens

        self.output_dim = input_dim 

    def forward(self, x):
        """
        Extract features from DINOv2 and project them to the required dimension.

        Args:
            x (Tensor): Input image (B, 3, 224, 224)

        Returns:
            Tensor: Extracted feature (B, out_dim)
        """
        # Extract features from DINOv2
        features_dict = self.dinov2.forward_features(x)

        # Extract CLS token (B, 384)
        cls_token = features_dict['x_norm_clstoken']

        if self.use_patch_tokens:
            # Extract patch tokens (B, 256, 384) and average them
            patch_tokens = features_dict['x_norm_patchtokens'].mean(dim=1)  # (B, 384)

            # Concatenate CLS token with avg patch token (B, 768)
            features = torch.cat([cls_token, patch_tokens], dim=-1)
        else:
            # Only use CLS token
            features = cls_token  # (B, 384)

        # Project to the required output dimension

        return features


def get_encoder(encoder_name, device):
    from torchvision import models
    if encoder_name == 'resnet50':
        # self.feature_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        feature_encoder = models.resnet50(pretrained=True)
                    # 去掉全局平均池化和全连接层
        # self.feature_encoder.avgpool = torch.nn.Identity()
        feature_encoder.fc = torch.nn.Identity() #  # (batch_size, 2048, 8, 8) with input (batch_size, 3, 256, 256) and no pooling layer； (batch_size, 2048, 1, 1) with input (batch_size, 3, 224, 224) and pooling layer

        # Define a custom forward pass to avoid implicit reshaping
        def custom_forward(x):
            x = feature_encoder.conv1(x)
            x = feature_encoder.bn1(x)
            x = feature_encoder.relu(x)
            x = feature_encoder.maxpool(x)

            x = feature_encoder.layer1(x)
            x = feature_encoder.layer2(x)
            x = feature_encoder.layer3(x)
            x = feature_encoder.layer4(x)
            
            # Global average pooling
            x = feature_encoder.avgpool(x) ## (batch_size, 2048, 1, 1)
            # squeeze the 2nd and 3rd dimensions
            x = torch.squeeze(x, 3)
            x = torch.squeeze(x, 2)
            
            # Return feature vector after avgpool (2048-dimensional vector per sample)
            return x

        # Replace the default forward pass
        feature_encoder.forward = custom_forward

        encoder_output_size = 2048
        feature_encoder.to(device)
    elif encoder_name == 'dinov2_vits14':
        feature_encoder = DINOv2FeatureEncoder(encoder_name='dinov2_vits14', use_patch_tokens=True, device=device)
        encoder_output_size = feature_encoder.output_dim
        ## 

    return feature_encoder, encoder_output_size
