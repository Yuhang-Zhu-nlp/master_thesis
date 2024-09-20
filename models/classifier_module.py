import torch
import torch.nn as nn
from transformers import AdamW
from typing import List, Tuple, Dict
import re

# model building
class classfier_module(nn.Module):
  def __init__(self,
               bert: nn.Module,
               hidden_size: int,
               label_size: int,
               pool_method: str='mean',
               layer: int=0) -> None:
    super(classfier_module, self).__init__()
    if pool_method == 'layer_weight_sum_cls':
      self.weight_para = nn.Parameter(torch.ones(1, 24)/24)
    elif pool_method == 'layer_weight_sum_word':
      self.weight_para = nn.Parameter(torch.ones(1, 1, 24))
      self.weight_norm = nn.Softmax(dim=2)
      self.scalar = nn.Parameter(torch.tensor([1], dtype=torch.float32))
    self.__bert = bert
    self.__layer = layer
    self.pool_method = pool_method
    # freeze bert model
    for para in self.__bert.parameters():
      para.requires_grad = False
    self.__classifier_head = nn.Sequential(
        nn.Linear(hidden_size, hidden_size*2),
        nn.ReLU(),
        nn.Linear(hidden_size*2, label_size)
    )

  def __mean_pooling(self, representations: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    '''
    # shape of representations: [batch_size, token_num, hidden_size]
    # shape of attention mask: [batch_size, token_num]
    # return vaule size: [batch_size, hidden_size]
    '''
    attention_mask4elements = attention_mask.unsqueeze(-1).expand(representations.shape)
    sum_representations = torch.sum(representations * attention_mask4elements, dim = 1)
    len_per_sentence = attention_mask4elements.sum(dim = 1)
    return sum_representations/len_per_sentence

  def __cls_weight_sum(self, BERT_out_layers: Tuple[torch.Tensor], layer: int=0) -> torch.Tensor:
    '''
    # shape of BERT_out_layers: ([batch_size, token_num, hidden_size] * 25)
    # return vaule size: [batch_size, hidden_size]
    '''
    if layer:
      return BERT_out_layers[layer][:,0]
    tensors = []
    for layer in BERT_out_layers:
      tensors.append(layer[:,0])
    stacked_tensor = torch.stack(tensors[1:], dim=1)
    weighted_tensor = self.weight_para.T.mul(stacked_tensor)
    return torch.sum(weighted_tensor, dim=1)

  def __word_weight_sum(self,
                        BERT_out_layers: Tuple[torch.Tensor],
                        attention_mask: torch.Tensor,
                        layer: int=0) -> torch.Tensor:
    if layer:
      return self.__mean_pooling(BERT_out_layers[layer][:, 1:-1], attention_mask[:, 2:])
    tensors = []
    for layer in BERT_out_layers:
      tensors.append(self.__mean_pooling(layer[:, 1:-1], attention_mask[:, 2:]))
    stacked_tensor = torch.stack(tensors[1:], dim=0)
    weighted_tensor = self.weight_norm(self.weight_para).T.mul(stacked_tensor)
    return torch.sum(weighted_tensor, dim=0)

  def forward(self,
              labels: torch.Tensor=[],
              position: torch.Tensor=[],
              offset_mapping: torch.Tensor=[],
              **batch) -> torch.Tensor:
    bert_out = self.__bert(**batch, output_hidden_states=True)
    if self.pool_method == 'mean':
      # compute pooled representation by mean all token representations
      representation = self.__mean_pooling(bert_out.last_hidden_state[:, 1:-1], batch['attention_mask'][:, 2:])
    elif self.pool_method == 'layer_weight_sum_cls':
      representation = self.__cls_weight_sum(bert_out.hidden_states, layer=self.__layer)
    elif self.pool_method == 'layer_weight_sum_word':
      representation = self.__word_weight_sum(bert_out.hidden_states, batch['attention_mask'], layer=self.__layer)
    else:
      raise ValueError(f'Unaccepted pooling method: {self.pool_method}')
    classifier_out = self.__classifier_head(representation).view(-1)
    return classifier_out

  def get_representation(self, **input) -> torch.Tensor:
    with torch.no_grad():
      bert_out = self.__bert(**input, output_hidden_states=True)
      if self.pool_method == 'mean':
        # compute pooled representation by mean all token representations
        representation = self.__mean_pooling(bert_out.last_hidden_state[:, 1:-1], input['attention_mask'][:, 2:])
      elif self.pool_method == 'layer_weight_sum_cls':
        representation = self.__cls_weight_sum(bert_out.hidden_states, layer=self.__layer)
      elif self.pool_method == 'layer_weight_sum_word':
        representation = self.__word_weight_sum(bert_out.hidden_states, input['attention_mask'], layer=self.__layer)
      else:
        raise ValueError(f'Unaccepted pooling method: {self.pool_method}')
    return representation

  def get_optim(self, lr_g1, lr, wd):
    param_groups = [
        {'params': self.weight_para, 'lr': lr_g1},
        {'params': self.__classifier_head.parameters()}
    ]
    k = AdamW(param_groups, lr=lr, weight_decay=wd)
    return k
