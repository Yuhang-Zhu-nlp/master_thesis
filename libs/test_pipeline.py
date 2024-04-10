import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import random
import json
import argparse
from .load_tokenizer import load_tokenizer_model
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, BertTokenizer


def test(device=None,
         tokenizer=None,
         dataset_path_pos: str='',
         dataset_path_neg: str='',
         random_seed: int='',
         model: nn.Module=None):
  labels = []
  preds = []
  model.eval()
  model.to(device)
  data_all:List[Tuple[str or int]] = []
  with open(dataset_path_pos, 'r', encoding='utf-8') as fp:
    for sentence in json.load(fp):
      data_all.append((sentence,1))
  random.seed(random_seed)
  with open(dataset_path_neg, 'r', encoding='utf-8') as fp:
    for sentence in random.sample(json.load(fp), len(data_all)):
      data_all.append((sentence, 0))
  for sent, label in data_all:
    input = tokenizer(sent, return_tensors='pt')
    out = nn.functional.sigmoid(model(input_ids=input['input_ids'].to(device), attention_mask=input['attention_mask'].to(device))).item()
    labels.append(label)
    preds.append(1 if out>=0.5 else 0)
  return labels, preds

def test_pipeline(config):
  pres = []
  recs = []
  f1s = []
  for seed in [3,4,5,6,7]:
    config['random_seed'] = seed
    labels, preds = test(**config)
    pre, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    pres.append(pre)
    recs.append(rec)
    f1s.append(f1)
  return {'f1': torch.mean(torch.tensor(f1s)).item(),
          'precision': torch.mean(torch.tensor(pres)).item(),
          'recall': torch.mean(torch.tensor(recs)).item()}