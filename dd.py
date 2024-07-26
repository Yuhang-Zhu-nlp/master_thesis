import json
import random
random.seed(20)
a,b=[], []
with open('data/zh_dev_have_neg.json', 'r', encoding='utf-8') as fp:
    a=json.load(fp)
with open('data/fi_dev_have_neg.json', 'r', encoding='utf-8') as fp:
    b=json.load(fp)
a=random.sample(a, 26)
a.extend(random.sample(b, 13))
#a.extend(b)
with open('zh_dev_have_neg1.json', 'w', encoding='utf-8') as fp:
    json.dump(a, fp, ensure_ascii=False)