import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import PeftModelForSeq2SeqLM


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


def get_mbart_encoder(llm_model):
    # 1. unwrap DDP
    if isinstance(llm_model, DDP):
        llm_model = llm_model.module

    # 2. LoRA-wrapped model (PeftModelForSeq2SeqLM)
    if isinstance(llm_model, PeftModelForSeq2SeqLM):
        base_model = llm_model.base_model # peft.tuners.lora.model.LoraModel            
        if hasattr(base_model, "model"):
            if isinstance(base_model.model, MBartForConditionalGeneration):
                return base_model.model.get_encoder()
        else:
            raise AttributeError(f"LoRA base_model.model is not MBartForConditionalGeneration. Got {type(base_model.model)}")


    # 3. Plain MBart
    if isinstance(llm_model, MBartForConditionalGeneration):
        return llm_model.get_encoder()

    raise AttributeError(f"Cannot find encoder in model of type {type(llm_model)}")

class YouTubeASLBaseline(nn.Module):
    def __init__(self, rgb_input_dim=512, pose_input_dim=138, freeze_llm=False, logger=None, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        llm_name = kwargs.get("llm_name", "mbart-large-50")  # Default to MBart50
        self.modality = kwargs.get("modality", "pose")  # Default to pose modality
        self.xD_pose = kwargs.get("xD_pose", "2D")  # Default to 2D pose
        
        self.llm_name = llm_name
        self.hidden_dim = kwargs.get("hidden_dim", 512)  # Default hidden dimension for pose features

        if "pose" in self.modality:
            self.pose_projector = nn.Linear(pose_input_dim, self.hidden_dim)  # Project pose features to 512 dim
            self.input_dim = self.hidden_dim  # Update input_dim to match pose projector output
            self.pose_temporal_encoder = TemporalEncoder(input_dim=self.hidden_dim)
            # self.pose_temporal_encoder = nn.Identity()
        else:
            self.pose_projector = nn.Identity()
            self.pose_temporal_encoder = nn.Identity()
            
        if 'rgb' in self.modality:
            self.rgb_temporal_encoder = TemporalEncoder(input_dim=rgb_input_dim)
            self.input_dim = rgb_input_dim
        else:
            self.rgb_temporal_encoder = nn.Identity()
            
        if 'pose' in self.modality and 'rgb' in self.modality:
            self.cross_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        
        if llm_name == "mbart-large-50":
            self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
            self.llm_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
            self.tokenizer.src_lang = "en_XX"
            self.tokenizer.tgt_lang = "en_XX"
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id["en_XX"]            

        else:
            raise ValueError(f"Unsupported llm_name: {llm_name}. Use 't5-base' or 'mbart-large-50'.")

        # Linear projection for pose → embedding
        self.llm_dim = self.llm_model.config.d_model  # usually 768
        self.fc = nn.Linear(self.input_dim, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim)


        if logger:
            logger.info(f"Loaded LLM model: {llm_name}", main_process_only=True)
            self.logger.info(f'self.llm_dim: {self.llm_dim}', main_process_only=True)
            logger.info(f"Modality: {self.modality}", main_process_only=True)
            logger.info(f"Freeze LLM parameters: {freeze_llm}", main_process_only=True)
            if freeze_llm:
                logger.info("Freezing LLM parameters.", main_process_only=True)
            else:
                logger.info("LLM parameters will be trained.")
        else:
            print(f"Loaded  {llm_name}")
            if freeze_llm:
                print("Freezing LLM parameters.")
            else:
                print("LLM parameters will be trained.")

        if freeze_llm:
            for param in self.llm_model.parameters():
                param.requires_grad = False


    def forward(self, lh = None, rh = None, body = None, face = None, 
                frame_feat = None, split = 'train', decoder_input_ids = None, 
                valid_frame_seq_mask = None,
                valid_pose_seq_mask = None,
                ):
        
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
            
            pose_features = self.pose_temporal_encoder(pose_features)  # Temporal encoding for pose features
            # print(f'pose_features.shape: {pose_features.shape}')
            attention_mask = valid_pose_seq_mask

        if "rgb" in self.modality:
            # print(f'frame_feat.shape: {frame_feat.shape}')
            rgb_features = self.rgb_temporal_encoder(frame_feat)  # Temporal encoding for RGB features
            # print(f'rgb_features.shape: {rgb_features.shape}')
            attention_mask = valid_frame_seq_mask
            
        if "pose" in self.modality and "rgb" in self.modality:
            # Cross-attention between pose and RGB features
            cross_attn_output, _ = self.cross_attention(pose_features, rgb_features, rgb_features)
            attention_mask = valid_pose_seq_mask | valid_frame_seq_mask

        elif "pose" in self.modality and not "rgb" in self.modality:
            cross_attn_output = pose_features
        elif not "pose" in self.modality and "rgb" in self.modality:
            cross_attn_output = rgb_features
        else:
            raise ValueError("At least one modality (pose or rgb) must be provided.")

        B, T, _ = cross_attn_output.shape

        # print('cross_attn_output', cross_attn_output)
        # Linear projection

        feat_embeds = self.fc(cross_attn_output)
        # print('feat_embeds', feat_embeds)
        # print("Before norm - mean:", feat_embeds.mean().item(), "std:", feat_embeds.std().item())

        feat_embeds = self.norm(feat_embeds)
        # print('feat_embeds', feat_embeds)
        # print("After norm - mean:", feat_embeds.mean().item(), "std:", feat_embeds.std().item())


        encoder = get_mbart_encoder(self.llm_model)


        encoder_outputs = encoder(
            inputs_embeds=feat_embeds,
            attention_mask=attention_mask
        )
        # print('encoder_outputs', encoder_outputs)
        
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
                forced_bos_token_id=self.decoder_start_token_id
            )

            decoded_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded_text, encoder_outputs.last_hidden_state

        else:
            raise ValueError("Invalid mode. Choose from ['train', 'val', 'test']")



if __name__ == "__main__":
    # Example usage
    batch_size, T, pose_input_dim = 3, 10, 138
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modality = 'pose'  # or 'rgb', or 'pose_rgb'
    model = YouTubeASLBaseline(pose_input_dim=pose_input_dim, modality=modality).to(device)
    frame_feat = torch.randn(batch_size, T, pose_input_dim).to(device)
    lh = torch.randn(batch_size, T, 21, 2).to(device)  # Left hand keypoints
    rh = torch.randn(batch_size, T, 21, 2).to(device)  # Right hand keypoints
    body = torch.randn(batch_size, T, 9, 2).to(device)  # Body keypoints
    face = torch.randn(batch_size, T, 18, 2).to(device)  # Face keypoints

    # # Training mode
    # decoder_input_ids = torch.randint(0, 250054, (batch_size, 20)).to(device)  # Example decoder input IDs
    # logits, encoder_hidden = model(lh=lh, rh=rh, body=body, face=face,
    #                                split="train", decoder_input_ids=decoder_input_ids)
    # print("Training logits shape:", logits.shape)
    # print("Encoder hidden shape:", encoder_hidden.shape)

    # Validation mode
    decoded_text, encoder_hidden = model(lh=lh, rh=rh, body=body, face=face,
                                         split="val")
    print("Decoded text:", decoded_text)
    print("Encoder hidden shape:", encoder_hidden.shape)