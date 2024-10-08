import sys
import os
if not os.path.dirname(os.path.dirname(__file__)) in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch
from libs.trainer import Trainer4classfier
from libs.dataset import dataset
from libs.load_tokenizer import load_tokenizer_model
from transformers import TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--pos_path_train", type=str, required=True, help="training set for positive instances")
parser.add_argument("--neg_path_train", type=str, required=True, help="training set for negative instances")
parser.add_argument("--neg_path_dev", type=str, required=True, help="development set for positive instances")
parser.add_argument("--pos_path_dev", type=str, required=True, help="development set for negative instances")
parser.add_argument("--model_name",
                    type=str,
                    required=False,
                    choices=['en_bert', 'xlm-roberta', 'ernie'],
                    default='xlm-roberta',
                    help="select model you want to train")
parser.add_argument("--pool_method",
                    type=str,
                    required=True,
                    choices=['mean', 'layer_weight_sum_cls', 'layer_weight_sum_word'],
                    help="select pool method")
parser.add_argument("--learning_rate", type=float, required=False, default=5e-4, help="hyperparameter set-up: learning rate")
parser.add_argument("--weight_decay", type=float, required=False, default=0.01, help="hyperparameter set-up: weight_decay")
parser.add_argument("--batch_size_train", type=int, required=False, default=32, help="batch size for training")
parser.add_argument("--batch_size_valid", type=int, required=False, default=32, help="batch size for validation")
parser.add_argument("--epoch", type=int, required=False, default=30, help="max epoch")
parser.add_argument("--warmup_steps", type=int, required=False, default=100, help="learning rate warm-up")
parser.add_argument("--checkpoints_dir", type=str, required=True, help="path to store checkpoints")
parser.add_argument("--output_dir", type=str, required=True, help="path to save model")
parser.add_argument("--is_wandb", action="store_true", required=False, help="whether store training in wandb")
parser.add_argument("--layer", type=int, required=False, default=0, help="use which layer of the model")
parser.add_argument("--name", type=str, required=False, default='', help="wandb run name")
args = parser.parse_args()

tokenizer, model = load_tokenizer_model(args.model_name, args.pool_method, args.layer)
if args.is_wandb:
    import wandb
    wandb.init(
        name=args.name,
        project="master_thesis_johan",
        config={
        "learning_rate": args.learning_rate,
        "architecture": args.model_name,
        "dataset": "UD",
        "epochs": args.epoch,
        }
    )
    train_args = TrainingArguments(
        args.checkpoints_dir,
        learning_rate=args.learning_rate,
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_valid,
        num_train_epochs=args.epoch,
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=8,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay)
else:
    train_args = TrainingArguments(
        args.checkpoints_dir,
        learning_rate=args.learning_rate,
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.batch_size_train,
        per_device_eval_batch_size=args.batch_size_valid,
        num_train_epochs=args.epoch,
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=8,
        overwrite_output_dir=True,
        weight_decay=args.weight_decay,
        report_to="none")

dataset.set_tokenizer(tokenizer)
dataset_train = dataset(args.pos_path_train, args.neg_path_train)
dataset_validation = dataset(args.pos_path_dev, args.neg_path_dev)
print(model)
print(f'size of training set: {len(dataset_train)}')
print(f'size of validation set: {len(dataset_validation)}')

if args.pool_method in ['layer_weight_sum_cls', 'layer_weight_sum_word'] and args.layer == 0:
    trainer = Trainer4classfier(model=model,
                            args=train_args,
                            train_dataset=dataset_train,
                            eval_dataset=dataset_validation,
                            data_collator=dataset.batch_collector_,
                            compute_metrics=Trainer4classfier.compute_metrics,
                            optimizers=(model.get_optim(0.01, args.learning_rate, args.weight_decay),None))
else:
    trainer = Trainer4classfier(model=model,
                            args=train_args,
                            train_dataset=dataset_train,
                            eval_dataset=dataset_validation,
                            data_collator=dataset.batch_collector_,
                            compute_metrics=Trainer4classfier.compute_metrics)
trainer.train()
if args.is_wandb:
    wandb.finish()
torch.save(model.state_dict(), args.output_dir)
