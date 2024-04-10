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
    bert_out = self.__bert(**batch, output_hidden_states=True)
    if self.pool_method == 'mean':
      # compute pooled representation by mean all token representations
      representation = self.__mean_pooling(bert_out.last_hidden_state[:, 1:-1], batch['attention_mask'][:, 2:])
    elif self.pool_method == 'layer_weight_sum_cls':
      representation = self.__cls_weight_sum(bert_out.hidden_states)
    elif self.pool_method == 'layer_weight_sum_word':
      representation = self.__word_weight_sum(bert_out.hidden_states, batch['attention_mask'])
    else:
      raise ValueError(f'Unaccepted pooling method: {self.pool_method}')
    classifier_out = self.__classifier_head(representation).view(-1)
    return classifier_out
