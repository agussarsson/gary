import torch
import torch.nn as nn
from transformers import AutoModel

class MultiHeadClassifier(nn.Module):
    def __init__(self, base_model_name: str, num_event_types: int, num_severities: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.event_head =  nn.Linear(hidden, num_event_types)
        self.severity_head = nn.Linear(hidden, num_severities)

    def forward(self, input_ids, attention_mask, event_labels=None, severity_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = self.dropout(cls)

        event_logits = self.event_head(x)
        severity_logits = self.severity_head(x)

        loss = None

        if event_labels is not None and severity_labels is not None:
            ce = nn.CrossEntropyLoss()
            loss_event = ce(event_logits, event_labels)
            loss_severity = ce(severity_logits, severity_labels)

            # weight the event more as it should be prioritized over severity
            loss = loss_event + 0.5 * loss_severity

        return {
            "loss": loss,
            "event_logits": event_logits,
            "severity_logits": severity_logits
        }