#! /bin/bash

path=/crex/proj/uppmax2020-2-2/yuhang/master_thesis_e2/master_thesis/venv

mkdir -p $path
module load conda
conda create -p $path/thesis python=3.12.2
export CONDA_ENVS_PATH=$path
conda activate thesis
pip install -r requirements.txt
