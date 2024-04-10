from datasets import load_dataset
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
import random
import torch
from transformers.tokenization_utils_base import BatchEncoding

sentence_max_length = 250
class dataset(Dataset):

  tokenizer = None

  def __init__(self, pos_path: str, neg_path: str) -> None:
    data = load_dataset('json', data_files={'pos': pos_path, 'neg': neg_path})
    self.data: List[Tuple[Dict[str, str], int]] = []
    for pos_d in data['pos']:
      self.data.append((pos_d, 1))
    random.seed(20)
    for neg_d in random.sample(list(data['neg']), len(data['pos'])):
      self.data.append((neg_d, 0))

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, item: int) -> Tuple[Dict[str, str], int]:
    return self.data[item]

  @staticmethod
  def set_tokenizer(tokenizer) -> None:
    dataset.tokenizer = tokenizer

  @staticmethod
  def batch_collector_(batch: List[Tuple[Dict[str, str], int]],) -> BatchEncoding:
    if dataset.tokenizer:
      sentences = dataset.tokenizer([data[0]['text'] for data in batch], return_tensors='pt',
                                     padding=True,truncation=True, max_length=250)
      sentences['labels'] = torch.tensor([data[1] for data in batch], dtype=torch.float)
      return sentences
    else:
      raise ValueError('please set tokenizer using dataset.set_tokenizer')
