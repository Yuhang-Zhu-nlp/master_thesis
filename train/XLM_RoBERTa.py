from classifier_module import classfier_module
from transformers import XLMRobertaModel
class xlm_roberta_pooling_representation(classfier_module):
  def __init__(self,
               pool_method: str='mean') -> None:
    bert = XLMRobertaModel.from_pretrained("xlm-roberta-large")
    super(xlm_roberta_pooling_representation, self).__init__(bert, 1024, 1, pool_method=pool_method)
