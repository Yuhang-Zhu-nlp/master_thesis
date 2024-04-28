#!/bin/bash -l

#SBATCH -A uppmax2020-2-2
#SBATCH -J johan_thesis
#SBATCH -t 10:00:00
#SBATCH -M snowy
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem 16g

for n in [1..24]
do
  python master_thesis/operation/train.py \
            --pos_path_test '/content/master_thesis/data/fr_train_fut_pos.json' \
            --neg_path_test '/content/master_thesis/data/fr_train_fut_neg.json' \
            --model_name 'xlm-roberta' \
            --pool_method 'layer_weight_sum_word' \
            --learning_rate 300 \
            --epoch 1500 \
            --out_dir /content/drive/MyDrive/master_thesis/it_roberta/it_roberta_word_layer${n}.pt\
            --layer ${n} \
            --model_path /proj/uppmax2020-2-2/yuhang/master_thesis/trained_model_en/fr_roberta_word_cmp_layer${n}.pt\
            -d 2
done