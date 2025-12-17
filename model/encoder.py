# model/encoder.py

import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token
        return outputs.last_hidden_state[:, 0]
