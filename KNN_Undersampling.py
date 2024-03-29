__author__ = 'ahmadauliawiguna'

__author__ = 'ahmadauliawiguna'
import matplotlib.pyplot as plt
import numpy as np
import urllib
import pprint
import operator
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.neighbors import NearestNeighbors
pp = pprint.PrettyPrinter(indent=4)

#-------------------------------------------------------LOAD DATASET--------------------------------------------
url = "http://localhost/dataset/winequality-white.data"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=";")
# separate the data from the target attributes
X = dataset[:,0:dataset.shape[1]-2] #ambil kolom dari kolom ke 0 sampai ke kolom 2 dari kanan
y = dataset[:,dataset.shape[1] - 1] #ambil kolom terakhir
y_class,index_class,jumlah_class  = np.unique(y,return_counts=True, return_index=True) #dapatkan target labelnya apa aja
min_index,min_class = min(enumerate(jumlah_class), key=operator.itemgetter(1)) #min_class jumlah class terkecil
max_index,max_class = max(enumerate(jumlah_class), key=operator.itemgetter(1)) #max_class jumlah class terbesar
#pisahkan dataset
arrays = {}
arrays_final = {}

print 'Jumlah Class : ',jumlah_class
print 'Jumlah Record di Class minoritas : ',min_class
print 'Jumlah Record di Class mayoritas : ',max_class

#looping class yang ada
for target in y_class:
    arrays[target] = []
    arrays_final[target] = [] #menampung data asli/ori

    for (index,target_label) in enumerate(y): #looping y, dapatkan target label dan recordnya
        if target_label==target: #jika record = target
            arrays[target].append(X[index])
            arrays_final[target].append(X[index]) #menampung data asli/ori
# looping array, cari yang jumlahnya lebih dari min_class
for (index,target_label) in enumerate(arrays):
    if len(arrays[target_label]) > min_class:
        k = len(arrays[target_label])
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(arrays[target_label])
        #cari kedekatan antar record
        distances, indices = nbrs.kneighbors(arrays[target_label])
        index_tetangga = indices[0][::-1] #balik array, karena mengambil item yang paling tidak bertetangga sejumlah min_class
        # index_tetangga = indices[0]
        index_tetangga = index_tetangga[0:min_class] #ambil index sejumlah min_class

        #kosongkan array
        arrays[target_label] = []

        #looping, ambil elemen
        for ambil in index_tetangga:
            arrays[target_label].append(arrays_final[target_label][ambil]) #AMBIL DATA BERDASAR TARGET DAN INDEX PALING TIDAK BERTETANGA

X_rknn = []
y_rknn = []
kolom = 0
baris = 0
for (index,target_label) in enumerate(arrays):
    for data in arrays[target_label]:
        X_rknn.append(data)
        y_rknn.append(target_label)
X_rknn = np.array(X_rknn) #convert normal array to numpy array
y_rknn = np.array(y_rknn) #convert normal array to numpy array

print 'Shape 1',X.shape
print 'Shape 2',X_rknn.shape
#-------------------------------------------------------BELUM DIPREPROCESSING----------------------
print '-------------------------------------------------------BELUM PREPROCESSING-------------------'
gaus_before = GaussianNB()

#kita split training dan testing 3 : 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
clf = gaus_before.fit(X_train,y_train)

#scoring
skor = gaus_before.score(X_test,y_test)
print('Score tanpa preprocessing : ',skor)
print('Sample Size tanpa preprocessing : ',X.shape)
scores = cross_val_score(clf , X_test,y_test, cv=3) #10fold cross validation
print("Accuracy tanpa preprocessing : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#-------------------------------------------------------SUDAH UNDER SAMLPLING----------------------
print '-------------------------------------------------------RANDOM UNDERSAMPING-------------------'
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


#-------------------------------------------------------OVER SAMLPLING----------------------
# Apply the random under-sampling
rus = RandomOverSampler(random_state=40)
X_resampled, y_resampled = rus.fit_sample(X, y)

X_oversampling = X_resampled
y_oversampling = y_resampled


print '-------------------------------------------------------kRNN UNDERSAMPING-------------------'
#-------------------------------------------------------kRNN UNDER SAMLPLING----------------------
gaus_before = GaussianNB()

#kita split training dan testing 3 : 10
X_train, X_test, y_train, y_test = train_test_split(X_rknn, y_rknn, test_size=.3, random_state=40)
clf = gaus_before.fit(X_train,y_train)

#scoring
skor = gaus_before.score(X_test,y_test)
print('Score : ',skor)
print('kRNN Size : ',X_rknn.shape)
print('Data Testing Size : ',X_test.shape)
scores = cross_val_score(clf , X_test,y_test, cv=3) #10fold cross validation
print("Accuracy kRNN : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print '-------------------------------------------------------kRNN UNDERSAMPING RANDOM OVERSAMPLING-------------------'
#-------------------------------------------------------kRNN UNDER SAMLPLING RANDOM----------------------
gaus_before = GaussianNB()

#kita split training dan testing 3 : 10
X_rknn = np.concatenate((X_rknn , X_oversampling))
y_rknn = np.concatenate((y_rknn , y_oversampling))
X_train, X_test, y_train, y_test = train_test_split(X_rknn, y_rknn, test_size=.3, random_state=40)
clf = gaus_before.fit(X_train,y_train)

#scoring
skor = gaus_before.score(X_test,y_test)
print('Score sebelum sampling : ',skor)
print('kRNN Size : ',X_rknn.shape)
print('Data Testing Size : ',X_test.shape)
scores = cross_val_score(clf , X_test,y_test, cv=3) #10fold cross validation
print("Accuracy kRNN Undersampling random Oversampling : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))