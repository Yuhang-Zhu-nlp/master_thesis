import json
import matplotlib.pyplot as plt

x = [i for i in range(1, 25)]
def get_data(path, prop):
    l = []
    with open(path, 'r') as fp:
        for r in json.load(fp):
            l.append(r[prop])
    return l
'''
l1 = get_data('it_fut_en.json', 'f1')
l2 = get_data('it_fut_sv.json', 'f1')
l3 = get_data('en_fut_sv.json', 'f1')
l4 = get_data('en_fut_it.json', 'f1')
'''
l1 = get_data('fr_cmp_en.json', 'f1')
l2 = get_data('fr_cmp_sv.json', 'f1')
l3 = get_data('en_cmp_sv.json', 'f1')
l4 = get_data('en_cmp_fr.json', 'f1')


plt.plot(x,l1,'s-',color = 'r',label="fr-en")
plt.plot(x,l2,'o-',color = 'g',label="fr-sv")
plt.plot(x,l3,'^-',color = 'b',label="en-sv")
plt.plot(x,l4,'b-',color = 'black',label="en-fr")
'''
plt.plot(x,l1,'s-',color = 'r',label="it-en")#s-:方形
plt.plot(x,l2,'o-',color = 'g',label="it-sv")#s-:方形
plt.plot(x,l3,'^-',color = 'b',label="en-sv")#s-:方形
plt.plot(x,l4,'d-',color = 'black',label="en-it")#s-:方形
'''
plt.xlabel("layer")
plt.ylabel("f1")#纵坐标名字
plt.legend(loc='best')
plt.xticks(range(0,25))
plt.title('trained on english')
plt.show()