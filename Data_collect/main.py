from Corpus_builder.Extract_pipeline import ExtractPipeline

tree_bank_path = '/Users/yuhangzhu/Downloads/Universal Dependencies 2/ud-treebanks-v2.13'
store_path = '../data'

pipeline = ExtractPipeline(tree_bank_path)
'''
En_fut_pos_train = pipeline.extract('English', 'train', 'future', is_pos=True)
En_fut_neg_train = pipeline.extract('English', 'train', 'future', is_pos=False)
En_fut_pos_dev = pipeline.extract('English', 'dev', 'future', is_pos=True)
En_fut_neg_dev = pipeline.extract('English', 'dev', 'future', is_pos=False)
En_fut_pos_test = pipeline.extract('English', 'test', 'future', is_pos=True)
En_fut_neg_test = pipeline.extract('English', 'test', 'future', is_pos=False)
It_fut_pos_test = pipeline.extract('Italian', 'test', 'future', is_pos=True)
It_fut_neg_test = pipeline.extract('Italian', 'test', 'future', is_pos=False)
It_fut_neg_train = pipeline.extract('Italian', 'train', 'future', is_pos=False)
It_fut_pos_train = pipeline.extract('Italian', 'train', 'future', is_pos=True)
It_fut_neg_dev = pipeline.extract('Italian', 'dev', 'future', is_pos=False)
It_fut_pos_dev = pipeline.extract('Italian', 'dev', 'future', is_pos=True)
Sw_fut_neg_train = pipeline.extract('Swedish', 'train', 'future', is_pos=False)
Sw_fut_pos_train = pipeline.extract('Swedish', 'train', 'future', is_pos=True)
pipeline.file_write_in(En_fut_pos_dev, f'{store_path}/en_dev_fut_pos.json', 'English', 'dev')
pipeline.file_write_in(En_fut_neg_dev, f'{store_path}/en_dev_fut_neg.json', 'English', 'dev')
pipeline.file_write_in(En_fut_pos_train, f'{store_path}/en_train_fut_pos.json', 'English', 'train')
pipeline.file_write_in(En_fut_neg_train, f'{store_path}/en_train_fut_neg.json', 'English', 'train')
pipeline.file_write_in(En_fut_pos_test, f'{store_path}/en_test_fut_pos.json', 'English', 'test')
pipeline.file_write_in(En_fut_neg_test, f'{store_path}/en_test_fut_neg.json', 'English', 'test')
pipeline.file_write_in(It_fut_pos_test, f'{store_path}/it_test_fut_pos.json', 'Italian', 'test')
pipeline.file_write_in(It_fut_neg_test, f'{store_path}/it_test_fut_neg.json', 'Italian', 'test')
pipeline.file_write_in(It_fut_pos_train, f'{store_path}/it_train_fut_pos.json', 'Italian', 'train')
pipeline.file_write_in(It_fut_neg_train, f'{store_path}/it_train_fut_neg.json', 'Italian', 'train')
pipeline.file_write_in(It_fut_pos_dev, f'{store_path}/it_dev_fut_pos.json', 'Italian', 'dev')
pipeline.file_write_in(It_fut_neg_dev, f'{store_path}/it_dev_fut_neg.json', 'Italian', 'dev')
pipeline.file_write_in(Sw_fut_pos_train, f'{store_path}/sw_train_fut_pos.json', 'Swedish', 'train')
pipeline.file_write_in(Sw_fut_neg_train, f'{store_path}/sw_train_fut_neg.json', 'Swedish', 'train')
'''
'''
It_cmp_pos_train = pipeline.extract('Italian', 'train', 'cmp', is_pos=True)
pipeline.file_write_in(It_cmp_pos_train, f'{store_path}/it_train_cmp_pos.json', 'Italian', 'train')
It_cmp_pos_dev = pipeline.extract('Italian', 'train', 'cmp', is_pos=True)
pipeline.file_write_in(It_cmp_pos_train, f'{store_path}/it_train_cmp_pos.json', 'Italian', 'train')
It_cmp_pos_train = pipeline.extract('Italian', 'train', 'cmp', is_pos=True)
pipeline.file_write_in(It_cmp_pos_train, f'{store_path}/it_train_cmp_pos.json', 'Italian', 'train')
print(len(It_cmp_pos_train))
'''
'''
En_cmp_pos_train = pipeline.extract('English', 'train', 'cmp', is_pos=True)
pipeline.file_write_in(En_cmp_pos_train, f'{store_path}/en_train_cmp_pos.json', 'English', 'train')
En_cmp_pos_dev = pipeline.extract('English', 'dev', 'cmp', is_pos=True)
pipeline.file_write_in(En_cmp_pos_dev, f'{store_path}/en_dev_cmp_pos.json', 'English', 'dev')
En_cmp_pos_test = pipeline.extract('English', 'test', 'cmp', is_pos=True)
pipeline.file_write_in(En_cmp_pos_test, f'{store_path}/en_test_cmp_pos.json', 'English', 'test')
En_cmp_neg_train = pipeline.extract('English', 'train', 'cmp', is_pos=False)
pipeline.file_write_in(En_cmp_neg_train, f'{store_path}/en_train_cmp_neg.json', 'English', 'train')
En_cmp_neg_dev = pipeline.extract('English', 'dev', 'cmp', is_pos=False)
pipeline.file_write_in(En_cmp_neg_dev, f'{store_path}/en_dev_cmp_neg.json', 'English', 'dev')
En_cmp_neg_test = pipeline.extract('English', 'test', 'cmp', is_pos=False)
pipeline.file_write_in(En_cmp_neg_test, f'{store_path}/en_test_cmp_neg.json', 'English', 'test')
'''
'''
Fr_cmp_pos_train = pipeline.extract('French', 'train', 'cmp', is_pos=True)
pipeline.file_write_in(Fr_cmp_pos_train, f'{store_path}/fr_train_cmp_pos.json', 'French', 'train')
Fr_cmp_pos_dev = pipeline.extract('French', 'dev', 'cmp', is_pos=True)
pipeline.file_write_in(Fr_cmp_pos_dev, f'{store_path}/fr_dev_cmp_pos.json', 'French', 'dev')
Fr_cmp_pos_test = pipeline.extract('French', 'test', 'cmp', is_pos=True)
pipeline.file_write_in(Fr_cmp_pos_test, f'{store_path}/fr_test_cmp_pos.json', 'French', 'test')
Fr_cmp_neg_train = pipeline.extract('French', 'train', 'cmp', is_pos=False)
pipeline.file_write_in(Fr_cmp_neg_train, f'{store_path}/fr_train_cmp_neg.json', 'French', 'train')
Fr_cmp_neg_dev = pipeline.extract('French', 'dev', 'cmp', is_pos=False)
pipeline.file_write_in(Fr_cmp_neg_dev, f'{store_path}/fr_dev_cmp_neg.json', 'French', 'dev')
Fr_cmp_neg_test = pipeline.extract('French', 'test', 'cmp', is_pos=False)
pipeline.file_write_in(Fr_cmp_neg_test, f'{store_path}/fr_test_cmp_neg.json', 'French', 'test')
'''
'''
En_have_pos_train = pipeline.extract('Finnish', 'train', 'have', is_pos=True)
pipeline.file_write_in(En_have_pos_train, f'{store_path}/fi_train_have_pos.json', 'Finnish', 'train')
En_have_pos_dev = pipeline.extract('Finnish', 'dev', 'have', is_pos=True)
pipeline.file_write_in(En_have_pos_dev, f'{store_path}/fi_dev_have_pos.json', 'Finnish', 'dev')
En_have_pos_test = pipeline.extract('Finnish', 'test', 'have', is_pos=True)
pipeline.file_write_in(En_have_pos_test, f'{store_path}/fi_test_have_pos.json', 'Finnish', 'test')
En_have_neg_train = pipeline.extract('Finnish', 'train', 'have', is_pos=False)
pipeline.file_write_in(En_have_neg_train, f'{store_path}/fi_train_have_neg.json', 'Finnish', 'train')
En_have_neg_dev = pipeline.extract('Finnish', 'dev', 'have', is_pos=False)
pipeline.file_write_in(En_have_neg_dev, f'{store_path}/fi_dev_have_neg.json', 'Finnish', 'dev')
En_have_neg_test = pipeline.extract('Finnish', 'test', 'have', is_pos=False)
pipeline.file_write_in(En_have_neg_test, f'{store_path}/fi_test_have_neg.json', 'Finnish', 'test')
print(len(En_have_pos_train))
'''
'''
Sw_fut_neg_train = pipeline.extract('Swedish', 'train', 'cmp', is_pos=False)
Sw_fut_pos_train = pipeline.extract('Swedish', 'train', 'cmp', is_pos=True)
pipeline.file_write_in(Sw_fut_pos_train, f'{store_path}/sw_train_cmp_pos.json', 'Swedish', 'train')
pipeline.file_write_in(Sw_fut_neg_train, f'{store_path}/sw_train_cmp_neg.json', 'Swedish', 'train')
'''
'''
Zh_fut_neg_test = pipeline.extract('Chinese', 'test', 'have', is_pos=False)
Zh_fut_pos_test = pipeline.extract('Chinese', 'test', 'have', is_pos=True)
pipeline.file_write_in(Zh_fut_pos_test, f'{store_path}/zh_test_have_pos.json', 'Chinese', 'test')
pipeline.file_write_in(Zh_fut_neg_test, f'{store_path}/zh_test_have_neg.json', 'Chinese', 'test')
'''
'''
e = pipeline.extract4vis('Italian', 'train')
pipeline.file_write_in4vis(e, f'{store_path}/it_test_cmp_vis.json',)
'''
'''
Sw_fut_pos_train = pipeline.extract('Swedish', 'test', 'future', is_pos=True)
print(len(Sw_fut_pos_train))
pipeline.file_write_in(Sw_fut_pos_train, f'{store_path}/sw_test_fut_pos.json', 'Swedish', 'test')
'''

pipeline.file_write_in4vis(pipeline.extract4vis_fut('Italian', 'train'),
                           f'{store_path}/it_train_fut_vis.json')