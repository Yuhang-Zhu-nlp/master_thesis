from datasets import load_dataset
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
import random
import torch
from transformers.tokenization_utils_base import BatchEncoding

sentence_max_length = 250
class dataset(Dataset):
  def __init__(self, pos_path: str, neg_path: str, num_neg: int) -> None:
    data = load_dataset('json', data_files={'pos': pos_path, 'neg': neg_path})
    self.data: List[Tuple[Dict[str, str], int]] = []
    for pos_d in data['pos']:
      self.data.append((pos_d, 1))
    for neg_d in random.sample(list(data['neg']), num_neg):
      self.data.append((neg_d, 0))

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, item: int) -> Tuple[Dict[str, str], int]:
    return self.data[item]

  @staticmethod
  def batch_collector_(batch: List[Tuple[Dict[str, str], int]]) -> BatchEncoding:
    sentences = tokenizer([data[0]['text'] for data in batch], return_tensors='pt',
                                  padding = True,truncation=True, max_length = sentence_max_length)
    sentences['labels'] = torch.tensor([data[1] for data in batch], dtype=torch.float)
    return sentences