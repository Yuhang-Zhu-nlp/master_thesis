#! /bin/bash

path=$0

mkdir -p $path
module load conda
conda create -p $path/thesis python=3.12.2
export CONDA_ENVS_PATH=$path
conda activate thesis
pip install -r requirements.txt
