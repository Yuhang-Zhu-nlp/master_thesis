from datasets import load_dataset
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
import random
import torch
from transformers.tokenization_utils_base import BatchEncoding

sentence_max_length = 250
class dataset_l(Dataset):

  tokenizer = None
  counter = 0

  def __init__(self, pos_path: str, neg_path: str) -> None:
    self.data: List[Tuple[Dict[str, str], int]] = []
    if isinstance(pos_path, list):
        assert len(pos_path)==len(neg_path)
        for i in range(len(pos_path)):
            self.set_data(pos_path[i], neg_path[i])

  def set_data(self, pos_path: str, neg_path: str):
    data = load_dataset('json', data_files={'pos': pos_path, 'neg': neg_path})
    for pos_d in data['pos']:
        self.data.append((pos_d, dataset_l.counter))
    random.seed(20)
    dataset_l.counter+=1
    for neg_d in random.sample(list(data['neg']), len(data['pos'])):
        self.data.append((neg_d, dataset_l.counter))
    dataset_l.counter+=1
  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, item: int) -> Tuple[Dict[str, str], int]:
    return self.data[item]

  @classmethod
  def set_tokenizer(cls, tokenizer) -> None:
    cls.tokenizer = tokenizer

  @staticmethod
  def batch_collector_(batch: List[Tuple[Dict[str, str], int]],) -> BatchEncoding:
    if dataset_l.tokenizer:
      sentences = dataset_l.tokenizer([data[0]['text'] for data in batch], return_tensors='pt',
                                     padding=True,truncation=True, max_length=250)
      sentences['labels'] = torch.tensor([data[1] for data in batch], dtype=torch.float)
      return sentences
    else:
      raise ValueError('please set tokenizer using dataset_l.set_tokenizer')
