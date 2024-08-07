from .Loader import CorpusLoader
import re
from typing import List, Tuple, Dict


def extract_trigger(sentence: list, trigger: list) -> list:
    return [token[0] for token in sentence if token[1].lower() in trigger]


def is_accept_have(position: int,
                   sentence: dict,
                   possessor: str,
                   possessee: str,
                   language: str = '') -> bool:
    counter = [1, 1]
    if language in ['English', 'Chinese','Swedish']:
        for ele in sentence['tree_lst']:
            if re.match(r'[^0-9]*' + str(position) + ':' + possessor, ele):
                counter[0] -= 1
            elif re.match(r'[^0-9]*' + str(position) + ':' + possessee, ele):
                counter[1] -= 1
    elif language in ['Finnish']:
        if sentence['lemma_lst'][int(position) - 1] != 'olla':
            return False
        position = -1
        for ele in sentence['tree_lst']:
            position = re.match(r'([0-9]+)' + ':' + possessor, ele)
            if position:
                position = position.group(1)
                if 'Case=Ade' in sentence['label_lst'][int(position)-1]:
                    counter[0] -= 1
                    break
        if position and position != -1:
            for i, ele in enumerate(sentence['tree_lst']):
                if re.match(r'[^0-9]*' + position + ':' + possessee, ele) and sentence['pos_tag'][i] == 'NOUN':
                    counter[1] -= 1
                    break
    return False if 1 in counter else True

def post_process_fut_en(id_list: List[str],
                         corpus: CorpusLoader) -> List[str]:
    counter = 0
    is_re: bool = True
    re_id_list: List[str] = []
    for id in id_list:
        for pos_tag in corpus[id]['pos_tag']:
            if pos_tag == 'VERB':
                counter += 1
            if counter > 1:
                is_re = False
                break
        if is_re:
            re_id_list.append(id)
        is_re = True
    return re_id_list

def extract_sentence_have(corpus: CorpusLoader,
                          trigger: list,
                          possessor: str = '',
                          possessee: str = '',
                          language: str = '',
                          mode: str = 'pos') -> list:
    assert mode in ['pos', 'neg']
    if not (possessee or possessor):
        raise ValueError('possessor and possessee can not be void string')
    elif language not in ['English', 'Chinese', 'Finnish']:
        raise ValueError(f'unaccepted language:{language}')
    else:
        re_id_lst = []
        re_id_lst_negative = []
        for sentence in corpus:
            possible_trigger_pos = extract_trigger(sentence['token_lst'], trigger)
            if 1 in [is_accept_have(position, sentence, possessor, possessee, language=language)
                     for position in possible_trigger_pos]:
                re_id_lst.append(sentence['sent_id'])
            else:
                re_id_lst_negative.append(sentence['sent_id'])
        return re_id_lst if mode == 'pos'\
            else re_id_lst_negative


def tense_extractor_future(corpus: CorpusLoader,
                           trigger: list,
                           language: str = '',
                           mode: str = 'pos') -> list:
    assert mode in ['pos', 'neg']
    re_id_lst_negative = []
    re_id_lst = []
    if language in ['English', 'Swedish']:
        for sentence in corpus:
            position_possible_aux = extract_trigger(sentence['token_lst'], trigger)
            if 1 in [1 if 'AUX' in sentence['pos_tag'][int(position)-1] else 0 for position in position_possible_aux]:
                re_id_lst.append(sentence['sent_id'])
            else:
                re_id_lst_negative.append(sentence['sent_id'])
    elif language in ['Italian']:
        for sentence in corpus:
            for i, label in enumerate(sentence['label_lst']):
                if 'Tense=Fut' in label and 'VERB' in sentence['pos_tag'][i]:
                    re_id_lst.append(sentence['sent_id'])
                else:
                    re_id_lst_negative.append(sentence['sent_id'])
    else:
        raise ValueError(f'unaccepted language: {language}')
    return re_id_lst if mode == 'pos'\
                    else re_id_lst_negative

def extract_comparison_fr(position: str, sentence: dict) -> bool:
    adj_position = re.match(r'([0-9]+)' + ':' + 'advmod', sentence['tree_lst'][int(position)-1])
    if (adj_position and sentence['pos_tag'][int(position)-1] == 'ADV' and
            sentence['pos_tag'][int(adj_position.group(1)) - 1] == 'ADJ'):
        noun_position = re.match(r'([0-9]+)' + ':' +
                                 'amod', sentence['tree_lst'][int(adj_position.group(1))-1])
        for i, lemma in enumerate(sentence['lemma_lst']):
            if lemma in ['il', 'le'] and sentence['pos_tag'][i] == 'DET':
                det_position = re.match(r'([0-9]+)' + ':' + 'det', sentence['tree_lst'][i])
                if det_position and (int(det_position.group(1)) == position or det_position.group(1) == adj_position.group(1)):
                    return False
                elif det_position and noun_position and det_position.group(1) == noun_position.group(1):
                    return False
    else:
        return False
    return True


def postprocess_cmp(sentence: dict, language: str) -> bool:
    if language == 'English':
        trigger = 'more'
    elif language == 'Swedish':
        trigger = 'mer'
    else:
        raise ValueError(f'Unaccepted language{language}')
    for position, token in sentence['token_lst']:
        if token.lower() == trigger:
            more_position = re.match(r'([0-9]+)' + ':' + 'advmod', sentence['tree_lst'][int(position)-1])
            if more_position and sentence['pos_tag'][int(more_position.group(1))-1] == 'ADJ':
                return False
    return True


def extract_comparison(corpus: CorpusLoader,
                       trigger: str,
                       language: str = '',
                       mode: str = 'pos') -> list:
    assert mode in ['pos', 'neg']
    re_id_lst = []
    re_id_lst_negative = []
    if language in ['English', 'Swedish']:
        for sentence in corpus:
            if 1 in [1 if token.endswith(trigger) and 'Degree=Cmp' in sentence['label_lst'][int(position) - 1] and 'ADJ' in sentence['pos_tag'][int(position) - 1]
                     else 0 for position, token in sentence['token_lst']]:
                re_id_lst.append(sentence['sent_id'])
            else:
                re_id_lst_negative.append(sentence['sent_id'])
        for i, sent_id in enumerate(re_id_lst):
            if not postprocess_cmp(corpus[sent_id], language):
                del re_id_lst[i]
        for i, sent_id in enumerate(re_id_lst_negative):
            if not postprocess_cmp(corpus[sent_id], language):
                del re_id_lst_negative[i]
    elif language in ['French', 'Italian']:
        for sentence in corpus:
            position_possible_tri = extract_trigger(sentence['token_lst'], [trigger])
            if 1 in [extract_comparison_fr(position, sentence) for position in position_possible_tri]:
                re_id_lst.append(sentence['sent_id'])
            else:
                re_id_lst_negative.append(sentence['sent_id'])
    else:
        raise ValueError(f'unaccepted language: {language}')
    return re_id_lst if mode == 'pos' else re_id_lst_negative

def extract_comparison_fr_vis(position: str, sentence: dict, language='French') -> bool:
    adj_position = re.match(r'([0-9]+)' + ':' + 'advmod', sentence['tree_lst'][int(position)-1])
    if (adj_position and sentence['pos_tag'][int(position)-1] == 'ADV' and
            sentence['pos_tag'][int(adj_position.group(1)) - 1] == 'ADJ'):
        noun_position = re.match(r'([0-9]+)' + ':' +
                                 'amod', sentence['tree_lst'][int(adj_position.group(1))-1])
        for i, lemma in enumerate(sentence['lemma_lst']):
            if lemma in ['il', 'le'] and sentence['pos_tag'][i] == 'DET':
                det_position = re.match(r'([0-9]+)' + ':' + 'det', sentence['tree_lst'][i])
                if det_position and (int(det_position.group(1)) == position or det_position.group(1) == adj_position.group(1)):
                    return False
                elif det_position and noun_position and det_position.group(1) == noun_position.group(1):
                    return False
    else:
        return False
    return ('plus ' if language == 'French' else 'più ') + (sentence['lemma_lst'][int(adj_position.group(1)) - 1] if '_' != sentence['lemma_lst'][int(adj_position.group(1)) - 1] else sentence['token_lst'][int(adj_position.group(1)) - 1][1])

def extract_comparison_vis(corpus: CorpusLoader,
                       trigger: str,
                       language: str = '') -> list:
    re_lst = []
    if language in ['English', 'Swedish']:
        for sentence in corpus:
            for position, token in sentence['token_lst']:
                if token.endswith(trigger) and 'Degree=Cmp' in sentence['label_lst'][int(position) - 1] and 'ADJ' in sentence['pos_tag'][int(position) - 1]:
                    if not token.lower() in re_lst:
                        re_lst.append(token.lower())
    elif language in ['French', 'Italian']:
        for sentence in corpus:
            position_possible_tri = extract_trigger(sentence['token_lst'], [trigger])
            for position in position_possible_tri:
                e_token = extract_comparison_fr_vis(position, sentence, language)
                if e_token and not e_token.lower() in re_lst:
                    re_lst.append(e_token.lower())
    else:
        raise ValueError(f'unaccepted language: {language}')
    return re_lst

def extract_fut_vis(corpus: CorpusLoader,
                           trigger: list,
                           language: str = '') -> list:
    re_lst = []
    if language in ['English', 'Swedish']:
        for sentence in corpus:
            position_possible_aux = extract_trigger(sentence['token_lst'], [trigger])
            for aux_p in position_possible_aux:
                verb_p = re.match(r'([0-9]+)' + ':' + 'aux', sentence['tree_lst'][int(aux_p) - 1])
                if verb_p and sentence['pos_tag'][int(verb_p.group(1))-1] == 'VERB' and sentence['token_lst'][int(verb_p.group(1))-1][1].lower() not in re_lst and sentence['token_lst'][int(verb_p.group(1))-1][1].lower() == sentence['lemma_lst'][int(verb_p.group(1))-1].lower():
                    re_lst.append(trigger+ ' ' + sentence['token_lst'][int(verb_p.group(1))-1][1].lower())
    elif language in ['Italian']:
        for sentence in corpus:
            for position, token in sentence['token_lst']:
                if 'Tense=Fut' in sentence['label_lst'][int(position) - 1] and 'VERB' in sentence['pos_tag'][int(position) - 1]:
                    if not token.lower() in re_lst:
                        re_lst.append(token.lower())
    else:
        raise ValueError(f'unaccepted language: {language}')
    return re_lst
