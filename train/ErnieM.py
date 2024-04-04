from transformers import ErnieMModel
from classifier_module import classfier_module
class erniem_pooling_representation(classfier_module):
  def __init__(self,
               pool_method: str='mean') -> None:
    bert = self.__bert = ErnieMModel.from_pretrained('susnato/ernie-m-large_pytorch')
    super(erniem_pooling_representation, self).__init__(bert, 1024, 1, pool_method=pool_method)
