import os


def english_finnish_extract(sentence: list) -> str:
    return sentence[-2]


def chinese_extract(sentence: list) -> str:
    return f'{sentence[6]}:{sentence[7]}'


def tree_ele_extract(language: str, sentence: list) -> str:
    if language in ['English', 'Finnish', 'Swedish', 'Italian']:
        return english_finnish_extract(sentence)
    elif language in ['Chinese', 'French']:
        return chinese_extract(sentence)
    else:
        raise ValueError(f'Unaccepted language:{language}')


class CorpusLoader:
    def __init__(self, language: str, filepath: str, mode: str = 'train', corpus_filter=()):
        assert mode in ['train', 'dev', 'test']
        treebank = [path for path in os.listdir(filepath) if language in path and path not in corpus_filter]
        self.filepaths = [f'{filepath}/{file}/{path}'
                          for file in treebank for path in os.listdir(f'{filepath}/{file}')
                          if mode in path and path.endswith('conllu')]
        self.language = language
        self.sent_lst = self.__file_open()

    def __file_open(self):
        re_file_lst = [{'token_lst': [], 'tree_lst': [], 'pos_tag': [],
                        'label_lst': [], 'lemma_lst': []}]
        for path in self.filepaths:
            with open(path, 'r', encoding='utf-8') as fp:
                for line in fp:
                    line2lst = line.split()
                    if not line2lst:
                        re_file_lst.append({'token_lst': [], 'tree_lst': [], 'pos_tag': [],
                                            'label_lst': [], 'lemma_lst': []})
                    elif line2lst[0] == '#' and line2lst[1] == 'sent_id':
                        re_file_lst[-1]['sent_id'] = line2lst[-1]
                    elif line2lst[0] == '#' and line2lst[1] == 'text':
                        re_file_lst[-1]['text'] = ' '.join(line2lst[3:])
                    elif line2lst[0].isdigit():
                        re_file_lst[-1]['lemma_lst'].append(line2lst[2])
                        re_file_lst[-1]['label_lst'].append(line2lst[5])
                        re_file_lst[-1]['token_lst'].append((line2lst[0], line2lst[1]))
                        re_file_lst[-1]['tree_lst'].append(tree_ele_extract(self.language, line2lst))
                        re_file_lst[-1]['pos_tag'].append(line2lst[3])
        if not re_file_lst[-1]['token_lst']:
            del re_file_lst[-1]
        return re_file_lst

    def __len__(self):
        return len(self.sent_lst)

    def __str__(self):
        return str(self.sent_lst)

    def __getitem__(self, item):
        if isinstance(item, str):
            for ele in self.sent_lst:
                if ele['sent_id'] == item:
                    return ele
        else:
            return self.sent_lst[item]
