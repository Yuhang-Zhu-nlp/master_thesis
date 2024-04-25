import torch
import torch.nn as nn
from transformers import Trainer, EvalPrediction
from transformers.tokenization_utils_base import BatchEncoding
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple, Dict

class Trainer4classfier(Trainer):
  def compute_loss(self,
                   model: nn.Module,
                   batch: BatchEncoding,
                   return_outputs: bool=False) -> Tuple[float, Dict[str, torch.Tensor]]:
    print(model.device)
    loss_func = nn.BCEWithLogitsLoss(reduction="mean")
    labels = batch['labels']
    model_output = model(**batch)
    loss = loss_func(model_output.view(-1), labels)
    return (loss, {'output':model_output}) if return_outputs else loss

  @staticmethod
  def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    predictions = nn.functional.sigmoid(torch.tensor(pred.predictions))
    labels = pred.label_ids
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'f1' : f1,
        'precision': prec,
        'recall': rec
    }
