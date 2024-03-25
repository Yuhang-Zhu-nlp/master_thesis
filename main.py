from Corpus_builder.Extract_pipeline import ExtractPipeline

tree_bank_path = '/Users/yuhangzhu/Downloads/Universal Dependencies 2/ud-treebanks-v2.13'
store_path = './data'
pipeline = ExtractPipeline(tree_bank_path)
En_fut_pos_train = pipeline.extract('English', 'train', 'future', is_pos=True)
En_fut_neg_train = pipeline.extract('English', 'train', 'future', is_pos=False)
En_fut_pos_dev = pipeline.extract('English', 'dev', 'future', is_pos=True)
En_fut_neg_dev = pipeline.extract('English', 'dev', 'future', is_pos=False)
pipeline.file_write_in(En_fut_pos_dev, f'{store_path}/en_dev_fut_pos.json', 'English', 'dev')
pipeline.file_write_in(En_fut_neg_dev, f'{store_path}/en_dev_fut_neg.json', 'English', 'dev')
pipeline.file_write_in(En_fut_pos_train, f'{store_path}/en_train_fut_pos.json', 'English', 'train')
pipeline.file_write_in(En_fut_neg_train, f'{store_path}/en_train_fut_neg.json', 'English', 'train')