from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
# from pathlib import Path
# import gary

@dataclass(frozen=True)
class LabelMaps:
    event2id: Dict[str, int]
    sev2id: Dict[str, int]

class ExceptionsDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_name: str, label_maps: LabelMaps, max_len: int=256):
        self.df = pd.read_csv(data_path)
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label_maps = label_maps
        self.max_len = max_len

        required = {"exercise_name", "session_exercises", "user_text", "event_type", "severity"}

        missing = required - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing columns in {data_path}: {missing}.")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        prompt = (
            f"""
            Exercise: {row['exercise_name']}
            Session exercises: {row['session_exercises']}
            User note: {row['user_text']}
            """
        )

        enc = self.tok(
            prompt,
            truncation=True,
            padding="max_length",
            max_length = self.max_len,
            return_tensors="pt"
        )

        event_id = self.label_maps.event2id[str(row["event_type"])]
        sev_id = self.label_maps.sev2id[str(row["severity"])]

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "event_labels": torch.tensor(event_id, dtype=torch.long),
            "severity_labels": torch.tensor(sev_id, dtype=torch.long)
        }