from Corpus_builder.Extract_pipeline import ExtractPipeline

tree_bank_path = '/Users/yuhangzhu/Downloads/Universal Dependencies 2/ud-treebanks-v2.13'
pipeline = ExtractPipeline(tree_bank_path)
print(len(pipeline.extract('English', 'train', 'cmp', is_pos=True)))
