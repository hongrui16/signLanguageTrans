import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, nhead=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T, D]
        return self.encoder(x)

class YouTubeASLBaseline(nn.Module):
    """Baseline T5 Model for Sign Language Translation (Pose → T5)"""
    def __init__(self, input_dim=256, freeze_llm=False, logger=None, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        llm_name = kwargs.get("llm_name", "mbart-large-50")  # Default to MBart50
        self.modality = kwargs.get("modality", "pose")  # Default to pose modality
        
        self.llm_name = llm_name
        self.hidden_dim = kwargs.get("hidden_dim", 256)  # Default hidden dimension for pose features
        
        if "pose" in self.modality:
            self.pose_projector = nn.Linear(input_dim, self.hidden_dim)  # Project pose features to 256 dim
            input_dim = self.hidden_dim  # Update input_dim to match pose projector output
        else:
            self.pose_projector = nn.Identity()
        
        if llm_name == "t5-base":
            self.tokenizer = T5Tokenizer.from_pretrained(llm_name)
            self.llm_model = T5ForConditionalGeneration.from_pretrained(llm_name)            
            self.decoder_start_token_id = self.tokenizer.pad_token_id

        elif llm_name == "mbart-large-50":
            self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
            self.llm_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
            self.tokenizer.src_lang = "en_XX"
            self.tokenizer.tgt_lang = "en_XX"
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id["en_XX"]            

        else:
            raise ValueError(f"Unsupported llm_name: {llm_name}. Use 't5-base' or 'mbart-large-50'.")

        # Linear projection for pose → embedding
        self.input_dim = input_dim
        self.llm_dim = self.llm_model.config.d_model  # usually 768
        self.fc = nn.Linear(input_dim, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim)

        self.temporal_encoder = TemporalEncoder(input_dim=self.input_dim)


        if logger:
            logger.info(f"Loaded LLM model: {llm_name}")
            logger.info(f"Pose dim: {input_dim} → LLM hidden dim: {self.llm_dim}")
            if freeze_llm:
                logger.info("Freezing LLM parameters.")
        else:
            print(f"Loaded  {llm_name}")
            print(f"Pose dim: {input_dim} → LLM hidden dim: {self.llm_dim}")
            if freeze_llm:
                print("Freezing LLM parameters.")

        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False

    def forward(self, lh = None, rh = None, body = None, face = None, 
                frame_feat = None, split = 'train', decoder_input_ids = None, attention_mask=None):
        
        """
        Args:
            lh, rh, body, face: (B, T, N, C) - Pose features
            split: 'train', 'val', or 'test'
            decoder_input_ids: (B, L)
            attention_mask: (B, T)
        Returns:
            train mode → (logits, encoder_hidden)
            test/val mode → (decoded text, encoder_hidden)
        """
        if "pose" in self.modality:
            if lh is None or rh is None or body is None or face is None:
                raise ValueError("All pose features (lh, rh, body, face) must be provided for pose modality.")
            pose_features = torch.cat([lh, rh, body, face], dim=2)
            # pose_features: (B, T, N, C) → (B, T, N*C)
            pose_features = pose_features.reshape(pose_features.size(0), pose_features.size(1), -1)
            
            pose_features = self.pose_projector(pose_features)  # Project to 256 dim
            
            if 'rgb' in self.modality:
                if frame_feat is None:
                    raise ValueError("frame_feat must be provided for rgb modality.")
                # Concatenate pose features with frame features
                assert pose_features.shape[0] == frame_feat.shape[0], "Batch size mismatch between pose features and frame features."
                assert pose_features.shape[1] == frame_feat.shape[1], "Time steps mismatch between pose features and frame features."
                
                input_feat = torch.cat([pose_features, frame_feat], dim=-1)  
            else:            
                input_feat = pose_features
        
        elif self.modality == "rgb":
            if frame_feat is None:
                raise ValueError("frame_feat must be provided for rgb modality.")
            input_feat = frame_feat
        else:
            raise ValueError(f"Unsupported modality: {self.modality}. Use 'pose', 'rgb', or 'pose_rgb'.")
            
        

        B, T, _ = input_feat.shape

        input_feat = self.temporal_encoder(input_feat)  # Before fc

        # Linear projection
        feat_embeds = self.fc(input_feat)
        feat_embeds = self.norm(feat_embeds)

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=input_feat.device)

        if hasattr(self.llm_model, "encoder"):
            encoder = self.llm_model.encoder
        else:
            encoder = self.llm_model.model.encoder  # e.g., for MBart

        encoder_outputs = encoder(
            inputs_embeds=feat_embeds,
            attention_mask=attention_mask
        )

        if split == "train":
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids must be provided in training mode")

            outputs = self.llm_model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids
            )
            return outputs.logits, encoder_outputs.last_hidden_state

        elif split in ["val", "test"]:
            generated_ids = self.llm_model.generate(
                encoder_outputs=encoder_outputs,
                max_length=64,
                num_beams=5,
                early_stopping=True,
                decoder_start_token_id=self.decoder_start_token_id
            )

            decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded_text, encoder_outputs.last_hidden_state

        else:
            raise ValueError("Invalid mode. Choose from ['train', 'val', 'test']")



if __name__ == "__main__":
    # Example usage
    batch_size, T, input_dim = 1, 10, 256
    model = YouTubeASLBaseline(input_dim=input_dim, modality = 'rgb')
    frame_feat = torch.randn(batch_size, T, input_dim)

    # Training mode
    decoder_input_ids = torch.randint(0, 1000, (batch_size, 20))  # Example decoder input IDs
    logits, encoder_hidden = model(frame_feat = frame_feat, split="train", decoder_input_ids=decoder_input_ids)
    print("Training logits shape:", logits.shape)
    print("Encoder hidden shape:", encoder_hidden.shape)

    # Validation mode
    decoded_text, encoder_hidden = model(frame_feat = frame_feat, split="val")
    print("Decoded text:", decoded_text)
    print("Encoder hidden shape:", encoder_hidden.shape)