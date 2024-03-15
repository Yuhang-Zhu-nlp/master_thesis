from Corpus_builder.Loader import CorpusLoader
from Corpus_builder.Extractor import (extract_sentence_have,
                                      tense_extractor_future,
                                      extract_comparison)

tree_bank_path = '/Users/yuhangzhu/Downloads/Universal Dependencies 2/ud-treebanks-v2.13'
En_UD = CorpusLoader('English', tree_bank_path, mode='test', corpus_filter=['UD_English-Atis', 'UD_English-ESLSpok',
                                                                             'UD_English-GENTLE', 'UD_English-GUMReddit',
                                                                             'UD_English-LinES', 'UD_English-ParTUT',
                                                                             'UD_English-Pronouns', 'UD_English-PUD'])
Zh_UD = CorpusLoader('Chinese', tree_bank_path, mode='dev', corpus_filter=['UD_Chinese-Beginner', 'UD_Chinese-CFL',
                                                                             'UD_Chinese-GSDSimp', 'UD_Chinese-HK',
                                                                             'UD_Chinese-PatentChar'])
Fi_UD = CorpusLoader('Finnish', tree_bank_path, mode='dev', corpus_filter=['UD_Finnish-FTB', 'UD_Finnish-OOD'])
Sw_UD = CorpusLoader('Swedish', tree_bank_path, mode='test')
It_UD = CorpusLoader('Italian', tree_bank_path, mode='dev', corpus_filter=['UD_Italian-MarkIT', 'UD_Italian-Old',
                                                                             'UD_Italian-ParlaMint', 'UD_Italian-TWITTIRO',
                                                                             'UD_Italian-ParTUT', 'UD_Italian-PoSTWITA', 'UD_Italian-PUD',
                                                                             'UD_Italian-Valico'])
Fr_UD = CorpusLoader('French', tree_bank_path, mode='dev', corpus_filter=['UD_French-FQB', 'UD_French-ParisStories',
                                                                          'UD_French-ParTUT', 'UD_French-PUD', 'UD_French-Rhapsodie',
                                                                          ])
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
print(len(Fr_com))
