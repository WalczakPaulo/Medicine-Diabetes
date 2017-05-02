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

def get_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    # download the file
    raw_data = urlopen(url)
    dataset = np.loadtxt(raw_data, delimiter=",")
    print(dataset.shape)
    X = dataset[:, 0:8]
    pca = PCA(n_components=4, whiten=True)
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