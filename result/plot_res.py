import json
import matplotlib.pyplot as plt
import numpy as np
c ='have'
x = np.array([i for i in range(0, 25)])
y = np.array([i for i in range(0,50,2)])
def get_data(path, prop):
    l = []
    with open(path, 'r') as fp:
        for r in json.load(fp):
            l.append(r[prop])
    return l

if c == 'fut':
    l1 = get_data('it_fut_en.json', 'f1')
    l2 = get_data('it_fut_sv.json', 'f1')
    l3 = get_data('en_fut_sv.json', 'f1')
    l4 = get_data('en_fut_it.json', 'f1')
    l5 = get_data('en_fut_en.json', 'f1')
    l6 = get_data('it_fut_it.json', 'f1')
    plt.plot(x, l1, 's-', color='r', label="it-en")  # s-:方形
    plt.plot(x, l2, 'o-', color='g', label="it-sv")  # s-:方形
    plt.plot(x, l3, '^-', color='b', label="en-sv")  # s-:方形
    plt.plot(x, l4, 'd-', color='black', label="en-it")  # s-:方形
elif c == 'cmp':
    l1 = get_data('fr_cmp_en.json', 'f1')
    l2 = get_data('fr_cmp_sv.json', 'f1')
    l3 = get_data('en_cmp_sv.json', 'f1')
    l4 = get_data('en_cmp_fr.json', 'f1')
    plt.plot(x, l1, 's-', color='r', label="fr-en")
    plt.plot(x, l2, 'o-', color='g', label="fr-sv")
    plt.plot(x, l3, '^-', color='b', label="en-sv")
    plt.plot(x, l4, 'd-', color='black', label="en-fr")
elif c == 'have':
    l1 = get_data('fi_have_en.json', 'f1')
    l2 = get_data('fi_have_zh.json', 'f1')
    l3 = get_data('en_have_fi.json', 'f1')
    l4 = get_data('en_have_zh.json', 'f1')
    plt.plot(x, l1, 's-', color='r', label="fi-en")
    plt.plot(x, l2, 'o-', color='g', label="fi-zh")
    plt.plot(x, l3, '^-', color='b', label="en-fi")
    plt.plot(x, l4, 'd-', color='black', label="en-zh")
'''
fig, ax = plt.subplots()
bar_width = np.array([0.45 for i in range(25)])
bars1 = ax.bar(y - bar_width, l3, bar_width, color='lightblue', edgecolor='black', label='en-sv')
bars2 = ax.bar(y + bar_width, l4, bar_width, color='lightgreen', edgecolor='black', label='en-it')
bars3 = ax.bar(y, l5, bar_width, color='red', edgecolor='black', label='en-en')
'''
plt.xlabel("layer")
plt.ylabel("f1")#纵坐标名字
plt.legend(loc='best')
plt.xticks(range(0,25))
#plt.xticks([i for i in range(0,50,2)], range(0,25))
plt.show()