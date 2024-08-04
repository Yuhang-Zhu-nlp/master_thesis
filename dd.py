import json
import random
random.seed(20)
a,b=[], []
with open('sw_test1_fut_pos.json', 'r', encoding='utf-8') as fp:
    a=json.load(fp)
with open('data/sw_test_fut_pos.json', 'r', encoding='utf-8') as fp:
    b=json.load(fp)
#a=random.sample(a, 26)
#a.extend(random.sample(b, 13))
a.extend(b)
with open('sw_test1_fut_pos.json', 'w', encoding='utf-8') as fp:
    json.dump(a, fp, ensure_ascii=False)