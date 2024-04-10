from transformers import ErnieMModel
from models.classifier_module import classfier_module
import torch
class erniem_pooling_representation(classfier_module):
  def __init__(self,
               pool_method: str='mean') -> None:
    bert = ErnieMModel.from_pretrained('susnato/ernie-m-large_pytorch')
    super(erniem_pooling_representation, self).__init__(bert, 1024, 1, pool_method=pool_method)

  def forward(self,
              labels: torch.Tensor = [],
              position: torch.Tensor = [],
              offset_mapping: torch.Tensor = [],
              **batch) -> torch.Tensor:
    batch['attention_mask'] = batch['input_ids'].clone().detach()
    batch['attention_mask'][batch['attention_mask'] != 1] = 2
    batch['attention_mask'][batch['attention_mask'] == 1] = 0
    batch['attention_mask'][batch['attention_mask'] == 2] = 1
    batch['labels'] = labels
    classifier_out = super(erniem_pooling_representation, self).forward(**batch)
    return classifier_out
