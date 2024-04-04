import argparse
import trainer
from dataset import dataset
from transformers import AutoTokenizer

output_dir = '../Checkpoints'
parser = argparse.ArgumentParser()
parser.add_argument("-pos_path_train")
parser.add_argument("-neg_path_train")
parser.add_argument("-neg_path_dev")
parser.add_argument("-pos_path_dev")
parser.add_argument("-model_name")
args = parser.parse_args()