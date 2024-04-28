import sys
import os

import matplotlib.pyplot as plt

if not os.path.dirname(os.path.dirname(__file__)) in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import numpy as np
from libs.load_tokenizer import load_tokenizer_model
from libs.dataset import dataset
from libs.t_sne import tsne_visualizer
from libs.opt_representations import get_representations

parser = argparse.ArgumentParser()
parser.add_argument("--pos_path_test", type=str, required=True, help="test set for positive instances")
parser.add_argument("--neg_path_test", type=str, required=True, help="test set for negative instances")
parser.add_argument("--model_name",
                    type=str,
                    required=True,
                    choices=['en_bert', 'xlm-roberta', 'ernie'],
                    help="select model you want to train")
parser.add_argument("--pool_method",
                    type=str,
                    required=True,
                    choices=['mean', 'layer_weight_sum_cls', 'layer_weight_sum_word'],
                    help="select pool method")
parser.add_argument("--layer", type=int, required=False, default=0, help="use which layer of the model")
parser.add_argument("-d", "--dimension", type=int, required=False, default=0, help="set embedding dimension")
parser.add_argument("--model_path", type=str, required=True, help="path to stored model")
parser.add_argument("--out_dir", type=str, required=True, help="path to stored results")
parser.add_argument("--epoch", type=int, required=False, default=1, help="max epoch")
parser.add_argument("--learning_rate", type=float, required=False, default=10, help="hyperparameter set-up: learning rate")
args = parser.parse_args()

tokenizer, model = load_tokenizer_model(args.model_name, args.pool_method, args.layer)
dataset.set_tokenizer(tokenizer)
dataset_test = dataset(args.pos_path_test, args.neg_path_test)
print(len(dataset_test))
vector2position = tsne_visualizer(dimension=args.dimension,
                                  epoch=args.epoch,
                                  lr=args.learning_rate)
representations, labels = get_representations(model, dataset_test)
embeddings = vector2position.fit(representations.numpy())
x_min, x_max = embeddings.min(0), embeddings.max(0)
embeddings_n = (embeddings - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
for i in range(len(labels)):
    plt.text(embeddings_n[i, 0],
             embeddings_n[i, 1],
             str(int(labels[i])), color='deepskyblue' if int(labels[i]) == 1 else 'red')
plt.xticks([])
plt.yticks([])
plt.show()
#plt.savefig(f'{args.out_dir}/layer{args.layer}.jpg')
