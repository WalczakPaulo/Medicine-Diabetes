import csv
import numpy as np
from sklearn.preprocessing import normalize

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
[['6', '148', '72', '35', '0', '33.6', '0.627', '50', '1'],...],

read_and_split_data() returns training and testing sets
'''


def read_data():
    dataset = []
    with open('data.csv') as csvfile:
        rows = csv.reader(csvfile)
        dataset = list(rows)
        dataset = datatype_change(dataset, float)
    return dataset


def datatype_change(item, func):
    if isinstance(item, list):
        return [datatype_change(x, func) for x in item]
    return func(item)


def read_and_split_data():
    dataset = read_data()
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    #dataset = dataset/dataset.max(axis=0)
    #dataset[:-1][:,-1] = normalize(dataset[:-1][:,-1], axis=0, norm='l1')
    training_size = int(0.7*len(dataset))
    test_x = list(dataset[:-training_size][:, :-1])
    test_x = normalize(test_x, axis=0, norm="l1")
    test_y = [[element, 1-element] for element in list(dataset[:-training_size][:, -1])]
    train_x = list(dataset[-training_size:][:, :-1])
    train_x = normalize(train_x, axis=0, norm="l1")
    train_y = [[element, 1-element] for element in list(dataset[-training_size:][:, -1])]
    sumlol = np.sum(train_x, axis=0)
    return train_x, train_y, test_x, test_y

read_and_split_data()