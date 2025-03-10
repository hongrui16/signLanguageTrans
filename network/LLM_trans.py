import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM

class SignLanguageLLM(nn.Module):
    """Large Language Model for Sign Language Translation"""
    def __init__(self, model_name="facebook/mbart-large-50", tokenizer = None):
        super(SignLanguageLLM, self).__init__()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Use Seq2SeqLM version

        # 解决 vocab size 不匹配的问题
        self.model.config.vocab_size = self.tokenizer.vocab_size
        self.model.resize_token_embeddings(self.tokenizer.vocab_size)


        # Linear layer to map 4C -> LLM embedding dim
        self.fc = nn.Linear(1024, self.model.config.hidden_size)  # hidden_size = 1024 for mbart-large-50
        self.norm = nn.LayerNorm(self.model.config.hidden_size)  # 添加归一化
        print(f"Model vocab size: {self.model.config.vocab_size}")
        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")


    def forward(self, sign_features, mode="train", decoder_input_ids=None, attention_mask=None):
        """
        Args:
            sign_features: (batch_size, T, 4C) - Aggregated pose features
            decoder_input_ids: (batch_size, seq_len) - Target token IDs (for training), optional
        Returns:
            If training (decoder_input_ids is not None):
                - logits: (batch_size, seq_len, vocab_size) - For loss computation
                - encoder_hidden: (batch_size, T, LLM_dim) - Encoder hidden representation
            If inference (decoder_input_ids is None):
                - translated_text: List[str] - Translated sentence
                - encoder_hidden: (batch_size, T, LLM_dim) - Encoder hidden representation
        """
        batch_size, T, C = sign_features.shape

        # 1️ Project to LLM dimension
        sign_features = self.fc(sign_features)  # (batch_size, T, LLM_dim)
        sign_features = self.norm(sign_features)  # 添加归一化

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, T).to(sign_features.device)


        # 2️ Encoder forward pass
        encoder_outputs = self.model.get_encoder()(
            inputs_embeds=sign_features,
            attention_mask=attention_mask
        )
        encoder_hidden = encoder_outputs.last_hidden_state


        # 3️ Decoder processing
        if mode == "train":                
            if decoder_input_ids is not None:
                # Training mode: Return logits for loss computation
                outputs = self.model(
                    encoder_outputs=encoder_outputs,  # Pass encoder outputs explicitly
                    decoder_input_ids=decoder_input_ids
                )
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                return logits, encoder_hidden
            else:
                raise ValueError("decoder_input_ids must be provided in training mode")
        elif mode in ["val", "test"]:
            input_length = encoder_outputs.last_hidden_state.shape[1]
            max_length = min(input_length * 2, 128)  # 限制最大长度，防止太长

            # Inference mode: Generate translated text
            generated_ids = self.model.generate(
                encoder_outputs=encoder_outputs,  # Use encoder outputs instead of inputs_embeds
                max_length=max_length,  # Adjust as needed
                num_beams=5,    # Beam search for better quality
                early_stopping=True
            )
            translated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return translated_text, encoder_hidden
        else:
            raise ValueError("Invalid mode. Choose from ['train', 'val', 'test']")

# Example usage
if __name__ == "__main__":
    # Dummy input
    batch_size, T, C = 2, 10, 1024
    sign_features = torch.randn(batch_size, T, C)

    # Initialize model
    model = SignLanguageLLM(model_name="facebook/mbart-large-50")

    # Inference mode
    translated_text, encoder_hidden = model(sign_features)
    print("Translated text:", translated_text)
    print("Encoder hidden shape:", encoder_hidden.shape)

    # Training mode
    decoder_input_ids = torch.randint(0, model.tokenizer.vocab_size, (batch_size, 8))  # Dummy target
    logits, encoder_hidden = model(sign_features, decoder_input_ids)
    print("Logits shape:", logits.shape)
    print("Encoder hidden shape:", encoder_hidden.shape)