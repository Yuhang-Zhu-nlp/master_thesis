import json
import random
random.seed(20)
a,b=[], []
with open('data/en_train_cmp_pos.json', 'r', encoding='utf-8') as fp:
    a=json.load(fp)
with open('data/fr_train_cmp_pos.json', 'r', encoding='utf-8') as fp:
    b=json.load(fp)
#a=random.sample(a, 69)
#a.extend(random.sample(b, 47))
a.extend(b)
with open('fr_train_cmp_pos1.json', 'w', encoding='utf-8') as fp:
    json.dump(a, fp, ensure_ascii=False)