__author__ = 'ahmadauliawiguna'
print 'Importing libraries.....'
import os
import subprocess as sp
import matplotlib.pyplot as plt
import numpy as np
import math
import urllib
import pprint
import operator
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

#random
from random import randint

#metrik
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

#normalisasi
from sklearn import preprocessing

# from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN

from texttable import Texttable

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
pp = pprint.PrettyPrinter(indent=4)
print 'Finish import.....'

min_max_scaler = preprocessing.MinMaxScaler()

#BEGIN
def print_menu() :
    for file in os.listdir("/Applications/XAMPP/htdocs/dataset"):
        if file.endswith(".data"):
            datasets.append(file)

    for (index_file,file) in enumerate(datasets):
        print '[',index_file,'] ',file

loop = True
while loop :
    datasets = []
    print 100 * '-'
    print_menu()
    state  = True
    chosen_file = ''
    while state:
        dataset = raw_input('Dataset [0-' + str(len(datasets)-1) +']: ')
        if dataset.lower() != 'exit':
            if (int(dataset) > (len(datasets)-1) or int(dataset) < 0):
                state  = True
            else:
                chosen_file = datasets[int(dataset)]
                state = False
        else:
            state = False
            loop = False
    if loop:
        print 'Dataset : ' , chosen_file
        the_k = 0.75
        state = True
        while state:
            #get separator dataset
            separator = raw_input('Pemisah kolom : ')
            if separator != ',' and separator != ';' and separator.lower() != 'exit':
                state  = True
            else :
                if separator.lower() == 'exit':
                    loop = False
                state = False

        state = True
        while state:
            #apakah mau dinormalisasi?
            normalisasi = raw_input('Normalisasi? (y/n) : ')
            if normalisasi != 'y' and normalisasi != 'n' and separator.lower() != 'exit':
                state  = True
            else :
                if separator.lower() == 'exit':
                    loop = False
                state = False

        the_k = raw_input('Prosentase k (0-1): ')
        the_k = float(the_k)
    if loop:
        data_table = [["Method/Param",    "Data Size (After Sampling)", "Data Training Size", "Data Testing Size","Akurasi", "F-Measure","Precision","Recall","TP","FP","TN","FN"]]#menampung hasil hitungan
        os.system('export TERM=clear')
        clear = lambda : os.system('clear')
        clear()
        #-------------------------------------------------------LOAD DATASET--------------------------------------------
        url = "http://localhost/dataset/" + chosen_file
        # download the file
        raw_data = urllib.urlopen(url)
        # load the CSV file as a numpy matrix
        dataset = np.loadtxt(raw_data, delimiter=separator)
        # separate the data from the target attributes
        X = dataset[:,0:dataset.shape[1]-2] #ambil kolom dari kolom ke 0 sampai ke kolom 2 dari kanan
        y = dataset[:,dataset.shape[1] - 1] #ambil kolom terakhir


        y_class,index_class,jumlah_class  = np.unique(y,return_counts=True, return_index=True) #dapatkan target labelnya apa aja
        min_index,min_class = min(enumerate(jumlah_class), key=operator.itemgetter(1)) #min_class jumlah class terkecil
        max_index,max_class = max(enumerate(jumlah_class), key=operator.itemgetter(1)) #max_class jumlah class terbesar


        X_awal = X #jaga-jaga aja, simpan X awal di X_awal
        #apakah di normalisasi?
        if normalisasi=='y':
            X = min_max_scaler.fit_transform(X)
        #pisahkan dataset
        arrays = {}
        arrays_final = {}

        print 'Jumlah Class : ',jumlah_class
        print 'Jumlah Record di Class minoritas : ',min_class
        print 'Jumlah Record di Class mayoritas : ',max_class

        #split dulu disini
        #X_train_krnn, X_test_krnn, y_train_krnn, y_test_krnn = train_test_split(X, y, test_size=.3, random_state=40)

        y_class,index_class,jumlah_class  = np.unique(y,return_counts=True, return_index=True) #dapatkan target labelnya apa aja
        min_index,min_class = min(enumerate(jumlah_class), key=operator.itemgetter(1)) #min_class jumlah class terkecil
        max_index,max_class = max(enumerate(jumlah_class), key=operator.itemgetter(1)) #max_class jumlah class terbesar


        #looping class yang ada
        for target in y_class:
            arrays[target] = []
            arrays_final[target] = [] #menampung data asli/ori

            for (index,target_label) in enumerate(y): #looping y hasil split sbg data training, dapatkan target label dan recordnya
                if target_label==target: #jika record = target
                    arrays[target].append(X[index])
                    arrays_final[target].append(X[index]) #menampung data asli/ori

        # looping array, cari yang jumlahnya kurang dari max_class
        for (index,target_label) in enumerate(arrays):
            if len(arrays[target_label]) < max_class:
                size_diambil = math.ceil(the_k * len(arrays[target_label])) #jumlah record yang mau dioversamplingkan
                size_class_sekarang = len(arrays[target_label]) #ukuran class s5ekarang

                k = len(arrays[target_label])
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(arrays[target_label])
                array_target_knn = arrays[target_label] #array yang mau di-kNN

                #cari kedekatan antar record
                distances, indices = nbrs.kneighbors(array_target_knn)
                ukuran_akhir = indices.shape[0] - 1;
                index_tetangga = indices[randint(0,ukuran_akhir)][::-1] #balik array, ambil urutan secara random,karena mengambil item yang paling tidak bertetangga sejumlah min_class
                #print index_tetangga
                # index_tetangga = indices[0]
                index_tetangga = index_tetangga[0:int(size_diambil)] #ambil index sejumlah size_diambil


                print 'Size class ',target_label,' k * jumlah record class minoritas : ',int(size_diambil)

                #kosongkan array
                #arrays[target_label] = []

                #looping sampai jumlah class sekarang >= class maksimal
                while len(arrays[target_label]) <= max_class:
                    for ambil in index_tetangga:
                        arrays[target_label].append(arrays_final[target_label][ambil]) #AMBIL DATA BERDASAR TARGET DAN INDEX PALING TIDAK BERTETANGA

                #jika size class sekarang melebihi jumlah class maksimal, kurangi saja
                arrays[target_label] = arrays[target_label][0:max_class]
        X_rknn = []
        y_rknn = []
        kolom = 0
        baris = 0

        #override X menjadi X_awal (jaga-jaga aja sih kalo pake normalisasi)
        if normalisasi=='y':
            X = X_awal

        print 100*'-'

        for (index,target_label) in enumerate(arrays):
            print 'Size class ',target_label,' sesudah dioversampling ',len(arrays[target_label])
            for data in arrays[target_label]:
                X_rknn.append(data)
                y_rknn.append(target_label)
        X_rknn = np.array(X_rknn) #convert normal array to numpy array
        y_rknn = np.array(y_rknn) #convert normal array to numpy array

        print 'Size Sebelum kRNN Oversampling',X.shape
        print 'Size Sesudah kRNN Oversampling',X_rknn.shape

        #-------------------------------------------------------TANPA OVER SAMLPLING----------------------
        gaus = GaussianNB()

        #kita split training dan testing 3 : 10
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=40)
        #eksekusi naive bayes
        clf = gaus.fit(X_train,y_train)

        #prediksi
        y_pred  = gaus.predict(X_test)
        #f-measure
        fm = f1_score(y_test,y_pred,average='micro')
        #akurasi
        akurasi = accuracy_score(y_test,y_pred)
        #presisi
        presisi = precision_score(y_test,y_pred)
        #recall
        recall = recall_score(y_test,y_pred)
        #confussion matrix
        # label = np.sort(np.unique(y_test))[::-1]
        label = np.unique(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,labels=label).ravel()

        #scoring
        skor = gaus.score(X_test,y_test)
        scores = cross_val_score(clf , X_test,y_test, cv=3,scoring='accuracy') #akurasi
        f1_macro = cross_val_score(clf , X_test,y_test, cv=3,scoring='f1_macro') #akurasi
        data_table.append(['No Resampling',str(X.shape),str(X_train.shape),str(X_test.shape), "%0.4f" % (akurasi*100),"%0.4f" % (fm*100),"%0.4f" % (presisi*100),"%0.4f" % (recall*100) , tp, fp , tn ,fn])


        #-------------------------------------------------------RANDOM OVER SAMLPLING----------------------
        # Apply the random under-sampling
        rus = RandomOverSampler(random_state=40)

        #declare Gaussian and fit with resampled dataset
        gaus = GaussianNB()


        #perform random oversampling terhadap data training
        X_resampled, y_resampled = rus.fit_sample(X, y)

        #split original dataset
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=.3, random_state=40)

        #fit Naive Bayes pada data training yg sudah diresampling
        clf = gaus.fit(X_train,y_train)

        #scoring
        skor = gaus.score(X_test,y_test)

        #prediksi
        y_pred  = gaus.predict(X_test)
        #presisi
        presisi = precision_score(y_test,y_pred)
        #recall
        recall = recall_score(y_test,y_pred)
        #akurasi
        akurasi = accuracy_score(y_test,y_pred)

        #confussion matrix
        # label = np.sort(np.unique(y_test))[::-1]
        label = np.unique(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,labels=label).ravel()

        #f-measure
        fm = f1_score(y_test,y_pred,average='micro')

        scores = cross_val_score(clf , X_test,y_test, cv=3,scoring='accuracy') #10fold cross validation
        f1_macro = cross_val_score(clf , X_test,y_test, cv=3,scoring='f1_macro') #10fold cross validation
        scores.shape
        data_table.append(['Random Oversampling',str(X_resampled.shape),str(X_train.shape),str(X_test.shape), "%0.4f" % (akurasi*100),"%0.4f" % (fm*100),"%0.4f" % (presisi*100),"%0.4f" % (recall*100) , tp, fp , tn ,fn ])


        # Apply the random under-sampling
        rus = RandomOverSampler(random_state=40)
        X_resampled, y_resampled = rus.fit_sample(X, y)

        X_oversampling = X_resampled
        y_oversampling = y_resampled


        #-------------------------------------------------------kRNN OVER SAMLPLING----------------------
        gaus = GaussianNB()

        #kita split training dan testing 3 : 10 (sudah di atas)
        # X_train, X_test, y_train, y_test = train_test_split(X_rknn, y_rknn, test_size=.3, random_state=40)
        X_train_krnn, X_test_krnn, y_train_krnn, y_test_krnn = train_test_split(X_rknn, y_rknn, test_size=.3, random_state=40)

        print 'Jumlah Data Training kRNN : ',X_train_krnn.shape[0]
        print 'Jumlah Data Testing kRNN  : ',X_test_krnn.shape[0]

        #fit Naive Bayes pada data yg sudah di-kRNN
        clf = gaus.fit(X_train_krnn,y_train_krnn)

        #prediksi menggunakan data test krnn
        y_pred  = gaus.predict(X_test_krnn)

        #f-measure
        fm = f1_score(y_test_krnn,y_pred,average='micro')
        #akurasi
        akurasi = accuracy_score(y_test_krnn,y_pred)
        #presisi
        presisi = precision_score(y_test_krnn,y_pred)
        #recall
        recall = recall_score(y_test_krnn,y_pred)
        #confussion matrix
        # label = np.sort(np.unique(y_test))[::-1]
        label = np.unique(y_test_krnn)
        tn, fp, fn, tp = confusion_matrix(y_test_krnn, y_pred,labels=label).ravel()

        #scoring
        skor = gaus.score(X_test_krnn,y_test_krnn)
        scores = cross_val_score(clf , X_test_krnn,y_test_krnn, cv=3,scoring='accuracy') #10fold cross validation
        f1_macro = cross_val_score(clf , X_test_krnn,y_test_krnn, cv=3,scoring='f1_macro') #10fold cross validation
        data_table.append(['kRNN Oversampling',str(X_rknn.shape),str(X_train_krnn.shape),str(X_test_krnn.shape), "%0.4f" % (akurasi*100),"%0.4f" % (fm*100),"%0.4f" % (presisi*100),"%0.4f" % (recall*100) , tp, fp , tn ,fn ])

        #-------------------------------------------------------SMOTE ----------------------
        gaus = GaussianNB()
        sm = SMOTE(random_state=42)

        #perform smote pada data training
        X_smote, y_smote= sm.fit_sample(X, y)

        #kita split training dan testing 3 : 10
        X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=.3, random_state=40)

        #fit Naive Bayes pada data training yg sudah di-SMOTE
        clf = gaus.fit(X_train,y_train)

        #prediksi
        y_pred  = gaus.predict(X_test)
        #f-measure
        fm = f1_score(y_test,y_pred,average='micro')
        #akurasi
        akurasi = accuracy_score(y_test,y_pred)
        #presisi
        presisi = precision_score(y_test,y_pred)
        #recall
        recall = recall_score(y_test,y_pred)
        #confussion matrix
        # label = np.sort(np.unique(y_test))[::-1]
        label = np.unique(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,labels=label).ravel()

        #scoring
        skor = gaus.score(X_test,y_test)
        scores = cross_val_score(clf , X_test,y_test, cv=3,scoring='accuracy') #10fold cross validation
        f1_macro = cross_val_score(clf , X_test,y_test, cv=3,scoring='f1_macro') #10fold cross validation
        data_table.append(['SMOTE',str(X_smote.shape),str(X_train.shape),str(X_test.shape), "%0.4f" % (akurasi*100),"%0.4f" % (fm*100),"%0.4f" % (presisi*100),"%0.4f" % (recall*100) , tp, fp , tn ,fn ])


        #-------------------------------------------------------(ADASYN) Adaptive Synthetic Sampling Approach for Imbalanced Learning ----------------------
        gaus = GaussianNB()
        sm = ADASYN(random_state=42)

        #perform ADASYN pada data training
        X_adasyn, y_adasyn= sm.fit_sample(X, y)

        #kita split training dan testing 3 : 10
        X_train, X_test, y_train, y_test = train_test_split(X_adasyn, y_adasyn, test_size=.3, random_state=40)

        #perform naive bayes pada data training yg sudah di-ADASYN
        clf = gaus.fit(X_train,y_train)

        #prediksi
        y_pred  = gaus.predict(X_test)
        #f-measure
        fm = f1_score(y_test,y_pred,average='micro')
        #akurasi
        akurasi = accuracy_score(y_test,y_pred)
        #presisi
        presisi = precision_score(y_test,y_pred)
        #recall
        recall = recall_score(y_test,y_pred)
        #confussion matrix
        # label = np.sort(np.unique(y_test))[::-1]
        label = np.unique(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,labels=label).ravel()

        #scoring
        skor = gaus.score(X_test,y_test)
        scores = cross_val_score(clf , X_test,y_test, cv=3,scoring='accuracy') #10fold cross validation
        f1_macro = cross_val_score(clf , X_test,y_test, cv=3,scoring='f1_macro') #10fold cross validation
        data_table.append(['ADASYN',str(X_adasyn.shape),str(X_train.shape),str(X_test.shape), "%0.4f" % (akurasi*100),"%0.4f" % (fm*100),"%0.4f" % (presisi*100),"%0.4f" % (recall*100) , tp, fp , tn ,fn ])

        #-------------------------------------------------------kRNN OVER SAMLPLING + SMOTE ----------------------
        gaus = GaussianNB()
        sm = SMOTE(random_state=42)
        X_smote, y_smote= sm.fit_sample(X, y)

        #gabungkan dataset hasil rKNN & SMOTE
        X_smote = np.concatenate((X_rknn , X_smote))
        y_smote = np.concatenate((y_rknn , y_smote))

        #kita split training dan testing 3 : 10
        X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=.3, random_state=40)
        clf = gaus.fit(X_train,y_train)


        #prediksi
        y_pred  = gaus.predict(X_test)
        #akurasi
        akurasi = accuracy_score(y_test,y_pred)
        #f-measure
        fm = f1_score(y_test,y_pred,average='micro')
        #presisi
        presisi = precision_score(y_test,y_pred)
        #recall
        recall = recall_score(y_test,y_pred)
        #confussion matrix
        # label = np.sort(np.unique(y_test))[::-1]
        label = np.unique(y_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred,labels=label).ravel()

        #scoring
        skor = gaus.score(X_test,y_test)
        scores = cross_val_score(clf , X_test,y_test, cv=3,scoring='accuracy') #10fold cross validation
        f1_macro = cross_val_score(clf , X_test,y_test, cv=3,scoring='f1_macro') #10fold cross validation
        #data_table.append(['kRNN Oversampling+SMOTE',str(X_smote.shape),str(X_train.shape), "%0.4f" % (akurasi*100),0,"%0.4f" % (fm*100),"%0.4f" % (presisi*100),"%0.4f" % (recall*100) , tp, fp , tn ,fn ])

        #DRAW TABLE-----------------------------------------------------------------------------------------------------------------------
        table = Texttable()
        # table.set_deco(Texttable.HEADER)
        table.set_cols_dtype(['t', 't',  't',  't', 't', 't','t',"t","t","t","t","t"]) # automatic
        table.set_cols_align(["l", "r", "r", "r", "r",'r',"r","r","r","r","r","r"])
        table.add_rows(data_table)
        print table.draw()
