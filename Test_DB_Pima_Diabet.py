__author__ = 'ahmadauliawiguna'
# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
import urllib

listone = [1, 2, 3]
listtwo = [4, 5, 6]
# print (listone + listtwo)

# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://goo.gl/j0Rvxq"
# download the file
raw_data = urllib.urlopen(url)
# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
# separate the data from the target attributes
X = dataset[:,0:7]
y = dataset[:,8]
# print(y.shape)
print(y[0:3])
print(X)
