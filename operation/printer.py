import sys
import os
if not os.path.dirname(os.path.dirname(__file__)) in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import argparse
from libs.load_tokenizer import load_tokenizer_model

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    type=str,
                    required=True,
                    choices=['en_bert', 'xlm-roberta', 'ernie'],
                    help="select model you want to evaluate")
parser.add_argument("--pool_method",
                    type=str,
                    required=True,
                    choices=['mean', 'layer_weight_sum_cls', 'layer_weight_sum_word'],
                    help="select pool method")
parser.add_argument("--model_path", type=str, required=True, help="the path that you store the model")
args = parser.parse_args()
tokenizer, model = load_tokenizer_model(args.model_name, args.pool_method, 0)
model.load_state_dict(torch.load(args.model_path))
print(model.weight_para)