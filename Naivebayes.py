__author__ = 'ahmadauliawiguna'
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler

print(iris.data.shape)

# load untuk tes akurasi
import numpy as np
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]


print('Skor akurasi',accuracy_score(y_true, y_pred))

x,y = make_classification(n_classes=2, class_sep=2,weights=[0.1,1.9],n_informative=3, n_redundant=1, flip_y=0
                          ,n_features=20,n_clusters_per_class=1, n_samples=100, random_state=10)

print('Datatype ',type(y))
print('Original dataset {}'.format(Counter(y)))
ros = RandomOverSampler(random_state=40)
x_res, y_res = ros.fit_sample(x,y)
print('Resampled dataset {}'.format(Counter(y_res)))
