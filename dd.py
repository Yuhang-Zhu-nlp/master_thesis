import json
import random
random.seed(20)
a,b=[], []
with open('data/en_dev_have_pos.json', 'r', encoding='utf-8') as fp:
    a=json.load(fp)
with open('data/zh_dev_have_pos.json', 'r', encoding='utf-8') as fp:
    b=json.load(fp)
a=random.sample(a, 83)
#a.extend(random.sample(b, 83))
a.extend(b)
with open('en_dev_have_pos1.json', 'w', encoding='utf-8') as fp:
    json.dump(a, fp, ensure_ascii=False)