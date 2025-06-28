import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


class YouTubeASLBaseline(nn.Module):
    """Baseline T5 Model for Sign Language Translation (Pose → T5)"""
    def __init__(self, input_dim=255, freeze_llm=False, logger=None, **kwargs):
        super().__init__()
        self.device = kwargs.get("device", "cpu")
        llm_name = kwargs.get("llm_name", "mbart-large-50")  # Default to MBart50
        self.llm_name = llm_name
        
        if llm_name == "t5-base":
            self.tokenizer = T5Tokenizer.from_pretrained(llm_name)
            self.model = T5ForConditionalGeneration.from_pretrained(llm_name)            
            self.decoder_start_token_id = self.tokenizer.pad_token_id

        elif llm_name == "mbart-large-50":
            self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
            self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
            self.tokenizer.src_lang = "en_XX"
            self.tokenizer.tgt_lang = "en_XX"
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id["en_XX"]            

        else:
            raise ValueError(f"Unsupported llm_name: {llm_name}. Use 't5-base' or 'mbart-large-50'.")

        # Linear projection for pose → embedding
        self.input_dim = input_dim
        self.llm_dim = self.model.config.d_model  # usually 768
        self.fc = nn.Linear(input_dim, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim)

        if logger:
            logger.info(f"Loaded LLM model: {model_name}")
            logger.info(f"Pose dim: {input_dim} → LLM hidden dim: {self.llm_dim}")
            if freeze_llm:
                logger.info("Freezing LLM parameters.")
        else:
            print(f"Loaded T5 model: {model_name}")
            print(f"Pose dim: {input_dim} → T5 hidden dim: {self.llm_dim}")
            if freeze_llm:
                print("Freezing LLM parameters.")

        if freeze_llm:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, lh, rh, body, face = None, split = 'train', decoder_input_ids = None,attention_mask=None):
        
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
        if not face is None:
            pose_features = torch.cat([lh, rh, body, face], dim=2)
        else:
            pose_features = torch.cat([lh, rh, body], dim=2)
        
        # pose_features: (B, T, N, C) → (B, T, N*C)
        pose_features = pose_features.reshape(pose_features.size(0), pose_features.size(1), -1)

        B, T, _ = pose_features.shape

        # Linear projection
        pose_embeds = self.fc(pose_features)
        pose_embeds = self.norm(pose_embeds)

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=pose_features.device)

        if hasattr(self.model, "encoder"):
            encoder = self.model.encoder
        else:
            encoder = self.model.model.encoder  # e.g., for MBart

        encoder_outputs = encoder(
            inputs_embeds=pose_embeds,
            attention_mask=attention_mask
        )

        if split == "train":
            if decoder_input_ids is None:
                raise ValueError("decoder_input_ids must be provided in training mode")

            outputs = self.model(
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids
            )
            return outputs.logits, encoder_outputs.last_hidden_state

        elif split in ["val", "test"]:
            generated_ids = self.model.generate(
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
    batch_size, T, input_dim = 1, 10, 255
    model = YouTubeASLBaseline(model_name="t5-base", input_dim=input_dim, freeze_llm=False)
    pose_features = torch.randn(batch_size, T, input_dim)

    # Training mode
    decoder_input_ids = torch.randint(0, 1000, (batch_size, 20))  # Example decoder input IDs
    logits, encoder_hidden = model(pose_features, split="train", decoder_input_ids=decoder_input_ids)
    print("Training logits shape:", logits.shape)
    print("Encoder hidden shape:", encoder_hidden.shape)

    # Validation mode
    decoded_text, encoder_hidden = model(pose_features, split="val")
    print("Decoded text:", decoded_text)
    print("Encoder hidden shape:", encoder_hidden.shape)