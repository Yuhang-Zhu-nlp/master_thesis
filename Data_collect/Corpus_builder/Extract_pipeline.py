from .Loader import CorpusLoader
from .Extractor import (extract_sentence_have,
                        tense_extractor_future,
                        extract_comparison,
                        extract_comparison_vis)
import json


corpus_filter = {'English': ['UD_English-Atis', 'UD_English-ESLSpok',
                             'UD_English-GENTLE', 'UD_English-GUMReddit',
                             'UD_English-LinES', 'UD_English-ParTUT',
                             'UD_English-Pronouns', 'UD_English-PUD'],
                 'Chinese': ['UD_Chinese-Beginner', 'UD_Chinese-CFL',
                             'UD_Chinese-GSDSimp', 'UD_Chinese-HK',
                             'UD_Chinese-PatentChar'],
                 'Finnish': ['UD_Finnish-FTB', 'UD_Finnish-OOD'],
                 'Swedish': [],
                 'Italian': ['UD_Italian-MarkIT', 'UD_Italian-Old',
                             'UD_Italian-ParlaMint', 'UD_Italian-TWITTIRO',
                             'UD_Italian-ParTUT', 'UD_Italian-PoSTWITA',
                             'UD_Italian-PUD', 'UD_Italian-Valico'],
                 'French': ['UD_French-FQB', 'UD_French-ParisStories',
                            'UD_French-ParTUT', 'UD_French-PUD', 'UD_French-Rhapsodie']}

def extract_position(sentence: dict, t_pos_tag: str='VERB') -> list:
    return [1 if pos_tag == t_pos_tag else 0 for i, pos_tag in enumerate(sentence['pos_tag'])]

class ExtractPipeline:
    def __init__(self, tree_bank_path: str):
        self.copora = [CorpusLoader(language, tree_bank_path, mode=mode, corpus_filter=corpus_filter[language])
                       for language in ['English', 'Chinese', 'Finnish', 'Swedish', 'Italian', 'French']
                       for mode in ['train', 'dev', 'test']]
        self.__langmode2id = {}
        counter = 0
        for language in ['English', 'Chinese', 'Finnish', 'Swedish', 'Italian', 'French']:
            for mode in ['train', 'dev', 'test']:
                self.__langmode2id[f'{language}_{mode}'] = counter
                counter += 1

    def extract4vis(self, language: str, mode: str):
        corpus = self.copora[self.__langmode2id[f'{language}_{mode}']]
        if language == 'English':
            trigger = 'er'
        elif language == 'Swedish':
            trigger = 're'
        elif language == 'Italian':
            trigger = 'più'
        else:
            trigger = 'plus'
        return extract_comparison_vis(corpus, trigger,
                                  language=language)

    def extract(self, language: str, mode: str, type: str, is_pos: bool = True) -> list:
        assert type in ['cmp', 'have', 'future']
        if type == 'cmp':
            return self.__cmp_extract(language, mode, is_pos)
        elif type == 'have':
            return self.__have_extract(language, mode, is_pos)
        else:
            return self.__future_extract(language, mode, is_pos)

    def __cmp_extract(self, language: str, mode: str, is_pos: bool = True) -> list:
        corpus = self.copora[self.__langmode2id[f'{language}_{mode}']]
        if language == 'English':
            trigger = 'er'
        elif language == 'Swedish':
            trigger = 're'
        elif language == 'Italian':
            trigger = 'più'
        else:
            trigger = 'plus'
        return extract_comparison(corpus, trigger,
                                  language=language,
                                  mode='pos' if is_pos else 'neg')

    def __have_extract(self, language: str, mode: str, is_pos: bool = True) -> list:
        corpus = self.copora[self.__langmode2id[f'{language}_{mode}']]
        if language == 'English':
            possessor, possessee, trigger = 'nsubj', 'obj', ['have', 'had', 'has']
        elif language == 'Chinese':
            possessor, possessee, trigger = 'nsubj', 'obj', ['有']
        else:
            possessor, possessee, trigger = 'cop:own', 'nsubj:cop', ['on']
        return extract_sentence_have(corpus, trigger,
                                     language=language,
                                     possessor=possessor,
                                     possessee=possessee,
                                     mode='pos' if is_pos else 'neg')

    def __future_extract(self, language: str, mode: str, is_pos: bool = True) -> list:
        corpus = self.copora[self.__langmode2id[f'{language}_{mode}']]
        trigger = ['']
        if language == 'English':
            trigger = ['will']
        elif language == 'Swedish':
            trigger = ['kommer']
        return tense_extractor_future(corpus, trigger,
                                      language=language,
                                      mode='pos' if is_pos else 'neg')

    def __getitem__(self, item):
        if isinstance(item, str) and item in self.__langmode2id:
            return self.copora[self.__langmode2id[item]]
        else:
            raise ValueError('item should obey the form language_mode')

    def __len__(self):
        return len(self.__langmode2id)

    def __str__(self):
        return str(list(self.__langmode2id.keys()))

    def file_write_in(self, id_list: list, path: str, language: str, mode: str):
        if not path.endswith('json'):
            raise ValueError('path should link to a json file')
        else:
            corpus = self.copora[self.__langmode2id[f'{language}_{mode}']]
            str_lst = [corpus[id]['text'] for id in id_list]
            with open(path, 'w', encoding='utf-8') as fp:
                json.dump(str_lst, fp, ensure_ascii=False)

    def file_write_in4vis(self, str_lst: list, path: str):
        if not path.endswith('json'):
            raise ValueError('path should link to a json file')
        else:
            with open(path, 'w', encoding='utf-8') as fp:
                json.dump(str_lst, fp, ensure_ascii=False)

    def extract_all_write_in(self, path: str):
        for language in ['English', 'Chinese', 'Finnish']:
            for mode in ['train', 'dev', 'test']:
                id_lst = self.__have_extract(language, mode, is_pos=True)
                self.file_write_in(id_lst, f'{path}/have_{language}_{mode}_pos', language, mode)
                id_lst = self.__have_extract(language, mode, is_pos=False)
                self.file_write_in(id_lst, f'{path}/have_{language}_{mode}_neg', language, mode)
        for language in ['English', 'Swedish', 'French']:
            for mode in ['train', 'dev', 'test']:
                id_lst = self.__cmp_extract(language, mode, is_pos=True)
                self.file_write_in(id_lst, f'{path}/cmp_{language}_{mode}_pos', language, mode)
                id_lst = self.__have_extract(language, mode, is_pos=False)
                self.file_write_in(id_lst, f'{path}/cmp_{language}_{mode}_neg', language, mode)
        for language in ['English', 'Swedish', 'Italian']:
            for mode in ['train', 'dev', 'test']:
                id_lst = self.__future_extract(language, mode, is_pos=True)
                self.file_write_in(id_lst, f'{path}/future_{language}_{mode}_pos', language, mode)
                id_lst = self.__have_extract(language, mode, is_pos=False)
                self.file_write_in(id_lst, f'{path}/future_{language}_{mode}_neg', language, mode)
