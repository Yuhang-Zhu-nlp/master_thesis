from sklearn import manifold
import matplotlib.pyplot as plt

class tsne_visualizer:
    def __init__(self, dimension:int= 2,
                        epoch:int= 2000,
                        lr: int= 300,
                        per:int =30):
        self.tsne = manifold.TSNE(n_components=dimension,
                     init='pca',
                     random_state=51,
                     n_iter=epoch,
                     learning_rate=lr,
                     verbose=1,
                    perplexity=per)

    def fit(self, data):
        return self.tsne.fit_transform(data)
