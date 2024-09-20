# Usage
## Environment Set-up
Here, we introduce how to set the environment of our experiments.
### Code Downloading
First of all, you need to download and enter this repository by the following codes:
```sh
git clone https://github.com/Yuhang-Zhu-nlp/master_thesis.git
cd master_thesis
```
### Environment Downloading
Then you need to download the environment by the following codes:
```sh
bash ./setup.sh path-to-store-environment
```

### Environment Loading
After downloading, the environment will be load automatically, but if you have downloaded the environment, you can simply run the following codes to load the environment directly.
```sh
module load conda
export CONDA_ENVS_PATH=path-to-store-environment
conda activate thesis
```
## Experiments
### Data Collection
Before running each experiment, you need to prepare data. The codes below work for data collection, and you need to download UD treebank first of all.
```sh
python ./operation/data_collect.py \
            --treebank_path path-to-UD-treebank \
            --output_path path-to-store-collected-data \
            --strategy the-strategy-you-want-to-use \
            --function the-function-you-want-to-use \
```
The above codes can be used to extract data for layer-separate strategy and visualization. The strategy has three possible values: layer-separate, layer-pooling, and visualization. Additionally, the function also has three possible values: fut, cmp, and have. However, before collecting data for layer-pooling strategy, you need to collect data for layer-separate strategy. Furthermore you need an extra parameter to collect data for layer-pooling strategy, and the codes become:
```sh
python ./operation/data_collect.py \
            --treebank_path path-to-UD-treebank \
            --output_path path-to-store-collected-data \
            --strategy the-strategy-you-want-to-use \
            --function the-function-you-want-to-use \
            --data_path the-path-to-where-you-store-data-for-layer-separate-strategy
```
## Training
To train the model, you can use the following codes:
```sh
python ./operation/train.py \
            --pos_path_train path-to-positive-training-instances\
            --neg_path_train path-to-negative-training-instances\
            --pos_path_dev path-to-positive-developing-instances\
            --neg_path_dev path-to-negative-training-instances\
            --pool_method method-to-represent-sentences\
            --learning_rate 5e-4\
            --weight_decay 0.01\
            --batch_size_train 32\
            --batch_size_valid 32\
            --epoch 30\
            --warmup_steps 100\
            --checkpoints_dir path-to-store-checkpoints\
            --output_dir path-to-store-the-final-model\
            --layer 1\
```
pool\_method has two possible values: mean and layer\_weight\_sum\_word. To run the experiments with layer-separate strategy, please set it as mean, and set it to layer\_weight\_sum\_word to run the experiments with layer-pooling strategy. Furthermore, to run the experiments with layer-pooling strategy, please set layer as 0.
## Visualization
We use t-SNE to visualize the representation of XLM-RoBERTa, and you can get the results of visualization by the following codes:
```sh
python ./operation/data_collect.py \
            --treebank_path path-to-UD-treebank \
            --output_path path-to-store-collected-data \
            --strategy the-strategy-you-want-to-use \
            --function the-function-you-want-to-use \
            --data_path the-path-to-where-you-store-data-for-layer-separate-strategy
```
## Evaluating
### F1 Score
To calculate the F1 score, please run the following codes:
```sh
python ./operation/evaluation.py \
            --pos_path_test path-to-positive-test-data \
            --neg_path_test path-to-negative-test-data \
            --model_name xlm-roberta \
            --pool_method mean \
            --model_path path-to-the-model \
            --layer 1
```
### Layer-importance Score
To calculate layer-importance score, you need to get the weights of the models first by the following codes:
```sh
python ./operation/printer.py \
            --model_name xlm-roberta \
            --pool_method layer_weight_sum_word \
            --model_path path-to-the-model \
```
Then you can calculate the layer-importance score manually by the equation in our paper.

   [@tjholowaychuk]: <http://twitter..md>
