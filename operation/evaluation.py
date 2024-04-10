import sys
import os
if not os.path.dirname(os.path.dirname(__file__)) in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import argparse
from libs.load_tokenizer import load_tokenizer_model
from libs.test_pipeline import test_pipeline


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument("--pos_path_test", type=str, required=True, help="test set for positive instances")
parser.add_argument("--neg_path_test", type=str, required=True, help="test set for negative instances")
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
tokenizer, model = load_tokenizer_model(args.model_name, args.pool_method)
model.load_state_dict(torch.load(args.model_path))
test_config = {
    'dataset_path_pos': args.pos_path_test,
    'dataset_path_neg': args.neg_path_test,
    'model': model,
    'tokenizer': tokenizer,
    'device': device
}
result = test_pipeline(test_config)
print()
