from Corpus_builder.Loader import CorpusLoader
from Corpus_builder.Extractor import (extract_sentence_have,
                                      tense_extractor_future,
                                      extract_comparison)

tree_bank_path = '/Users/yuhangzhu/Downloads/Universal Dependencies 2/ud-treebanks-v2.13'
En_UD = CorpusLoader('English', tree_bank_path, mode='train')
Zh_UD = CorpusLoader('Chinese', tree_bank_path, mode='train')
Fi_UD = CorpusLoader('Finnish', tree_bank_path, mode='train', corpus_filter=['UD_Finnish-FTB'])
Sw_UD = CorpusLoader('Swedish', tree_bank_path, mode='train')
It_UD = CorpusLoader('Italian', tree_bank_path, mode='train')
Fr_UD = CorpusLoader('French', tree_bank_path, mode='dev')
En_have = extract_sentence_have(En_UD, ['have', 'had', 'has'],
                                language='English',
                                possessor='nsubj', possessee='obj')
Zh_have = extract_sentence_have(Zh_UD, ['有'],
                                language='Chinese',
                                possessor='nsubj', possessee='obj')
Fi_have = extract_sentence_have(Fi_UD, ['on'],
                                language='Finnish',
                                possessor='cop:own', possessee='nsubj:cop')
En_fut = tense_extractor_future(En_UD, ['will'], language='English')
Sw_fut = tense_extractor_future(Sw_UD, ['kommer'], language='Swedish')
It_fut = tense_extractor_future(It_UD, [''], language='Italian')
En_com = extract_comparison(En_UD, 'er', language='English')
Sw_com = extract_comparison(Sw_UD, 're', language='Swedish')
Fr_com = extract_comparison(Fr_UD, 'plus', language='French')
print(len(Zh_have))
