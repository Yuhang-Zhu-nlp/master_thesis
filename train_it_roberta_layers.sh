for n in [1..24]
do
  python master_thesis/operation/train.py \
            --pos_path_train '/content/master_thesis/data/it_train_fut_pos.json' \
            --neg_path_train '/content/master_thesis/data/it_train_fut_neg.json' \
            --pos_path_dev '/content/master_thesis/data/it_dev_fut_pos.json' \
            --neg_path_dev '/content/master_thesis/data/it_dev_fut_neg.json' \
            --model_name 'xlm-roberta' \
            --pool_method 'layer_weight_sum_word' \
            --learning_rate 5e-4 \
            --weight_decay 0.01 \
            --batch_size_train 32 \
            --batch_size_valid 32 \
            --epoch 30 \
            --warmup_steps 100 \
            --checkpoints_dir /content/drive/MyDrive/checkpoints/iii${n}\
            --output_dir /content/drive/MyDrive/master_thesis/it_roberta/it_roberta_word_layer${n}.pt\
            --layer ${n} \
            --is_wandb
done