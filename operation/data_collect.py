import sys
import os
if not os.path.dirname(os.path.dirname(__file__)) in sys.path:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import json
import random
import re
from typing import List
from Data_collect.Corpus_builder.Extract_pipeline import ExtractPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--treebank_path",
                    type=str,
                    required=True,
                    help="Path to store UD treebank")
parser.add_argument("--output_path",
                    type=str,
                    required=True,
                    help="Path to store collected data")
parser.add_argument("--strategy",
                    type=str,
                    required=True,
                    choices=['layer-separate', 'layer-pooling', 'visualization'],
                    help="Select which experiment you want to run")
parser.add_argument("--function",
                    type=str,
                    required=True,
                    choices=['fut', 'cmp', 'have'],
                    help="Select the function")
parser.add_argument("--data_path",
                    type=str,
                    required=False,
                    default='',
                    help="Path to data")
args = parser.parse_args()
pipeline = ExtractPipeline(args.treebank_path)
func2lang = {
    'fut': ['English', 'Swedish', 'Italian'],
    'cmp': ['English', 'Swedish', 'French'],
    'have': ['English', 'Chinese', 'Finnish']
}
lang2id = {
    'English': 'en',
    'Swedish': 'sv',
    'Chinese': 'zh',
    'French': 'fr',
    'Finnish': 'fi'
}
m_lst = ['train', 'dev', 'test']


def collect_layer_separate(function: str) -> None:
    for lang in func2lang[function]:
        for m in m_lst:
            data = pipeline.extract(lang, m, function, is_pos=False)
            pipeline.file_write_in(data,
                                   '{}/{}_{}_{}_neg.json'.format(args.output_path, lang2id[lang], m, function),
                                   lang, m)
            data = pipeline.extract(lang, m, function, is_pos=True)
            pipeline.file_write_in(data,
                                   '{}/{}_{}_{}_pos.json'.format(args.output_path, lang2id[lang], m, function),
                                   lang, m)


def merge_pos(function: str,
          language1: str,
          language2: str) -> List[int]:
    nums = []
    for m in m_lst:
        try:
            l1, l2 = []
            with open('{}/{}_{}_{}_pos.json'.format(args.data_path, lang2id[language1], m, function), 'r',
                      encoding='utf-8') as fp:
                l1 = json.load(fp)
            with open('{}/{}_{}_{}_pos.json'.format(args.data_path, lang2id[language2], m, function), 'r',
                      encoding='utf-8') as fp:
                l2 = json.load(fp)
            num = min(len(l1), len(l2))
            nums.append(num)
            bump = random.sample(l1, num)
            bump.extend(random.sample(l2, num))
            with open('{}/{}_{}_{}_{}_pos.json'.format(args.output_path, language1, language2, m, function), 'w', encoding='utf-8') as fp:
                json.dump(bump, fp, ensure_ascii=False)
        except FileNotFoundError as e:
            file_path = re.match('.*No such file or directory: (.*)$', str(e))
            if file_path:
                file_path = file_path.group(1)
                raise FileNotFoundError(
                    f'fail to find the file: {file_path}, please check whether you have run the data collection for layer-separate strategy of the function {function}.\n'+\
                    'Note: You need to run data collection for layer-separate strategy before collecting data for layer-pooling strategy.'
                )
            else:
                raise
    return nums

def merge_neg(function: str,
          language1: str,
          language2: str,
          nums: List[int]):
    assert len(nums) == 3
    for i, m in enumerate(m_lst):
        try:
            l1, l2 = []
            with open('{}/{}_{}_{}_neg.json'.format(args.data_path, lang2id[language1], m, function), 'r',
                      encoding='utf-8') as fp:
                l1 = json.load(fp)
            with open('{}/{}_{}_{}_neg.json'.format(args.data_path, lang2id[language2], m, function), 'r',
                      encoding='utf-8') as fp:
                l2 = json.load(fp)
            num = nums[i]
            bump = random.sample(l1, num)
            bump.extend(random.sample(l2, num))
            with open('{}/{}_{}_{}_{}_neg.json'.format(args.output_path, language1, language2, m, function), 'w', encoding='utf-8') as fp:
                json.dump(bump, fp, ensure_ascii=False)
        except FileNotFoundError as e:
            file_path = re.match('.*No such file or directory: (.*)$', str(e))
            if file_path:
                file_path = file_path.group(1)
                raise FileNotFoundError(
                    f'fail to find the file: {file_path}, please check whether you have run the data collection for layer-separate strategy of the function {function}.\n' + \
                    'Note: You need to run data collection for layer-separate strategy before collecting data for layer-pooling strategy.'
                )
            else:
                raise

def collect_layer_pooling(function: str) -> None:
    if not args.data_path:
        raise ValueError("Please set data_path")
    else:
        for lang in func2lang[function][1:]:
            nums = merge_pos(func2lang[function][0], lang)
            merge_neg(func2lang[function][0], lang, nums)


def collect_layer_visualization(function: str):
    def collect_fut_vis(function: str):
        data = pipeline.extract4vis_fut(lang, 'train')
        pipeline.file_write_in4vis(data, '{}/{}_{}_vis.json'.format(args.output_path, lang2id[lang], function))
    def collect_cmp_vis(function: str):
        data = pipeline.extract4vis_cmp(lang, 'train')
        pipeline.file_write_in4vis(data, '{}/{}_{}_vis.json'.format(args.output_path, lang2id[lang], function))
    def collect_have_vis(function: str):
        data = pipeline.extract4vis_have(lang, 'train')
        pipeline.file_write_in4vis(data, '{}/{}_{}_vis.json'.format(args.output_path, lang2id[lang], function))
    if function == 'fut':
        for lang in func2lang[function]:
            collect_fut_vis(function)
    elif function == 'cmp':
        for lang in func2lang[function]:
            collect_cmp_vis(function)
    else:
        for lang in func2lang[function]:
            collect_have_vis(function)


if args.strategy == 'layer-separate':
    collect_layer_separate(args.function)
elif args.strategy == 'layer-pooling':
    collect_layer_pooling(args.function)
else:
    collect_layer_visualization(args.function)
