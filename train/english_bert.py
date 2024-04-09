from transformers import BertModel
from classifier_module import classfier_module

class english_bert(classfier_module):
    def __init__(self,
                pool_method: str = 'mean') -> None:
        bert = BertModel.from_pretrained('bert-large-cased')
        super(english_bert, self).__init__(bert, 1024, 1, pool_method=pool_method)
