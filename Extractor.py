from Loader import Corpus_loader
import re


def extract_trigger(sentence : list, trigger : list) -> list:
    return [token[0] for token in sentence if token[1].lower() in trigger]


def is_accept_have(position : int, tree : list) -> bool:
    counter = [1,1]
    for ele in tree:
        if re.match(r'[^0-9]?' + str(position) + ':nsubj', ele):
            counter[0] -= 1
        elif re.match(r'[^0-9]?' + str(position) + ':obj', ele):
            counter[1] -= 1
    return False if 1 in counter else True


def extract_sentence_have(Corpus : Corpus_loader, trigger : list) -> list:
    re_id_lst = []
    for sentence in Corpus:
        possible_trigger_pos = extract_trigger(sentence['token_lst'], trigger)
        if 1 in [is_accept_have(position, sentence['tree_lst']) for position in possible_trigger_pos]:
            re_id_lst.append(sentence['sent_id'])
    return re_id_lst