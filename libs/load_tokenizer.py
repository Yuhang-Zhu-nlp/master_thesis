from transformers import AutoTokenizer, BertTokenizer
from models.XLM_RoBERTa import xlm_roberta_pooling_representation
from models.english_bert import english_bert
from models.ErnieM import erniem_pooling_representation


def load_tokenizer_model(model_name: str, pool_method: str, layer: int):
    if model_name == 'en_bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
        model = english_bert(pool_method=pool_method, layer=layer)
    elif model_name == 'xlm-roberta':
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large')
        model = xlm_roberta_pooling_representation(pool_method=pool_method, layer=layer)
    elif model_name == 'ernie':
        tokenizer = AutoTokenizer.from_pretrained('susnato/ernie-m-large_pytorch')
        model = erniem_pooling_representation(pool_method=pool_method, layer=layer)
    else:
        raise ValueError(f'unaccepted model {model_name}')
    return tokenizer, model
