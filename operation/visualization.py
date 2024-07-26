import sys
import os

import matplotlib.pyplot as plt

if not os.path.dirname(os.path.dirname(__file__)) in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import numpy as np
import torch
from libs.load_tokenizer import load_tokenizer_model
from libs.dataset_vis import dataset_l
from libs.t_sne import tsne_visualizer
from libs.opt_representations import get_representations

parser = argparse.ArgumentParser()
parser.add_argument('--pos_path_test', nargs='+', help='test set for positive instances', required=True)
parser.add_argument('--labels', nargs='+', help='set labels', required=True)
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
parser.add_argument("-d", "--dimension", type=int, required=False, default=0, help="set embedding dimension")
parser.add_argument("--model_path", type=str, required=True, help="path to stored model")
parser.add_argument("--out_dir", type=str, required=True, help="path to stored results")
parser.add_argument("--epoch", type=int, required=False, default=1, help="max epoch")
parser.add_argument("--perplexity", type=int, required=False, default=30, help="perplexity of t-sne")
parser.add_argument("--learning_rate", type=float, required=False, default=10, help="hyperparameter set-up: learning rate")
args = parser.parse_args()

dataset_test = dataset_l(args.pos_path_test)
print(len(dataset_test))
vector2position = tsne_visualizer(dimension=args.dimension,
                                  epoch=args.epoch,
                                  lr=args.learning_rate,
                                  per=args.perplexity)
if args.dimension == 3:
    fig = plt.figure(figsize=(10, 5))
    ax = []
    for index, layer in enumerate(range(1,25)):
        i = index // 6
        j = index % 6
        ax.append(fig.add_subplot(4, 6, (i, j), projection='3d'))
        tokenizer, model = load_tokenizer_model(args.model_name, args.pool_method, layer)
        model.load_state_dict(torch.load(f'{args.model_path}{layer}.pt'))
        dataset_l.set_tokenizer(tokenizer)
        representations, labels = get_representations(model, dataset_test)
        embeddings = vector2position.fit(representations.numpy())
        x_min, x_max = embeddings.min(0), embeddings.max(0)
        embeddings_n = (embeddings - x_min) / (x_max - x_min)
        label_a_e = {}
        for i, l in enumerate(labels):
            label_a_e[l] = label_a_e.get(l, [])
            label_a_e[l].append((embeddings_n[i, 0],
                                 embeddings_n[i, 1],
                                 embeddings_n[i, 2]))
        for l in label_a_e:
            p_X, p_Y, p_Z = zip(*label_a_e[l])
            ax[index].scatter(list(p_X),
                              list(p_Y),list(p_Z), s=2, color=plt.cm.Spectral(int(l) * 30), label=args.labels[int(l)])
        sca, leg = ax[index].get_legend_handles_labels()
    fig.legend(sca, leg, loc='right')
    plt.savefig(f'{args.out_dir}/layer{args.dimension}.jpg', dpi=600)
else:
    fig, ax = plt.subplots(4, 6, sharex='col', sharey='row')
    for index, layer in enumerate(range(1,25)):
      tokenizer, model = load_tokenizer_model(args.model_name, args.pool_method, layer)
      model.load_state_dict(torch.load(f'{args.model_path}{layer}.pt'))
      dataset_l.set_tokenizer(tokenizer)
      representations, labels = get_representations(model, dataset_test)
      embeddings = vector2position.fit(representations.numpy())
      x_min, x_max = embeddings.min(0), embeddings.max(0)
      embeddings_n = (embeddings - x_min) / (x_max - x_min)
      label_a_e = {}
      for i, l in enumerate(labels):
        label_a_e[l] = label_a_e.get(l,[])
        label_a_e[l].append((embeddings_n[i, 0],
                             embeddings_n[i, 1]))
      i = index//6
      j = index%6
      for l in label_a_e:
        p_X, p_Y = zip(*label_a_e[l])
        ax[i, j].scatter(list(p_X),
                 list(p_Y), color=plt.cm.Spectral(int(l)*30), s=2, label = args.labels[int(l)])
      sca, leg = ax[i, j].get_legend_handles_labels()
    fig.legend(sca, leg, loc='right')
    plt.savefig(f'{args.out_dir}/layer{args.dimension}.jpg', dpi=600)
