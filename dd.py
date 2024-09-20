import json
import random
random.seed(20)
a,b=[], []
with open('data/fi_train_have_neg.json', 'r', encoding='utf-8') as fp:
    a=json.load(fp)
with open('data/en_train_have_neg.json', 'r', encoding='utf-8') as fp:
    b=json.load(fp)
a=random.sample(a, 126)
a.extend(random.sample(b, 126))
#a.extend(b)
'''
with open('fi_train_have_neg1.json', 'w', encoding='utf-8') as fp:
    json.dump(a, fp, ensure_ascii=False)
'''
with open('zh_test1_have_pos.json', 'r', encoding='utf-8') as fp:
    print(len(json.load(fp)))