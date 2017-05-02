import matplotlib.pyplot as plt
import numpy as np
import urllib
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


url = "http://localhost/dataset/glass.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:dataset.shape[1]-2] #ambil kolom dari kolom ke 0 sampai ke kolom 2 dari kanan
y = dataset[:,dataset.shape[1] - 1] #ambil kolom terakhir


#-------------------------------------------------------BELUM UNDER SAMLPLING----------------------
gaus_before = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
clf = gaus_before.fit(X_train,y_train)
#scoring
skor = gaus_before.score(X_test,y_test)
print('Score sebelum sampling : ',skor)
print('Sample Size sebelum sampling : ',X.shape)
scores = cross_val_score(clf , X_test,y_test, cv=3) #10fold cross validation
print("Accuracy sebelum sampling : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#-------------------------------------------------------SUDAH UNDER SAMLPLING----------------------
# Apply the random under-sampling
rus = RandomUnderSampler(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)

#declare Gaussian and fit with resampled dataset
gaus = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.3, random_state=40)
clf = gaus.fit(X_train,y_train)

#scoring
skor = gaus.score(X_test,y_test)


print('Score : ',skor)
print('Data Testing Size : ',X_test.shape)
print('Undersampling Size : ',X_resampled.shape)
scores = cross_val_score(clf , X_resampled,y_resampled, cv=3) #10fold cross validation
scores.shape
print("Accuracy sesudah sampling : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


