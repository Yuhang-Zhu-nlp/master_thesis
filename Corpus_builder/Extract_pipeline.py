from .Loader import CorpusLoader
from .Extractor import (extract_sentence_have,
                        tense_extractor_future,
                        extract_comparison)


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
        trigger = ''
        if language == 'English':
            trigger = 'er'
        elif language == 'Swedish':
            trigger = 're'
        else:
            trigger = 'plus'
        return extract_comparison(corpus, trigger,
                                  language=language,
                                  mode='pos' if is_pos else 'neg')

    def __have_extract(self, language: str, mode: str, is_pos: bool = True) -> list:
        corpus = self.copora[self.__langmode2id[f'{language}_{mode}']]
        possessor, possessee, trigger = '', '', []
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
        tense_extractor_future(corpus, trigger,
                               language=language,
                               mode='pos' if is_pos else 'neg')