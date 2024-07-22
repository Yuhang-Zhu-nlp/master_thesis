import json
import random
random.seed(20)
a,b=[], []
with open('data/en_dev_fut_neg.json', 'r', encoding='utf-8') as fp:
    a=json.load(fp)
with open('data/sw_dev_fut_neg.json', 'r', encoding='utf-8') as fp:
    b=json.load(fp)
a=random.sample(a, 24)
a.extend(random.sample(b, 24))
#a.extend(b)
with open('sw_dev_fut_neg1.json', 'w', encoding='utf-8') as fp:
    json.dump(a, fp, ensure_ascii=False)