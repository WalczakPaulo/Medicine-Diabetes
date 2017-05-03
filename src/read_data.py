import tensorflow as tf
import numpy as np
from urllib.request import urlopen
import skflow
from sklearn.preprocessing import Normalizer
from sklearn import datasets, metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
# Pima Indians Diabetes dataset (UCI Machine Learning Repository)

'''
Data Format:
1. Number of times pregnant.
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1) 

read_data() returns list of lists:
..where original data ...
[['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1'],...],
..is split into training and testing sets, and normalized to be efficiently used by neural network. Also PCA reduction is used

'''
def get_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # download the file
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",")
    print(dataset.shape)
    X = dataset[:, 0:8]
    pca = PCA(n_components=6, whiten=True)
    pca.fit(X)
    y = dataset[:, 8]
    X= preprocessing.scale(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print('test')
    y_train = [[element, 1 - element] for element in y_train]
    y_test = [[element, 1 - element] for element in y_test]
    return X_train, y_train, X_test, y_test

#get_data()
