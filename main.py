from Loader import Corpus_loader
from Extractor import extract_sentence_have

tree_bank_path = '/Users/yuhangzhu/Downloads/Universal Dependencies 2/ud-treebanks-v2.13'
En_UD = Corpus_loader('English',
                      tree_bank_path,
                      mode='dev')
Zh_UD = Corpus_loader('Chinese',
                      tree_bank_path,
                      mode='dev')
# Fi_UD = Corpus_loader('Finnish',
# tree_bank_path)

print(len(extract_sentence_have(En_UD, ['have', 'had', 'has'])))
print(len(extract_sentence_have(Zh_UD, ['有'])))
