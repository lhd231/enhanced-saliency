import numpy as np
from sklearn import linear_model
import sys
import os

data_path= sys.argv[1]
labels_path = sys.argv[2]

data_test_begin = int(sys.argv[3])
data_test_end = int(sys.argv[4])

data = np.load(data_path)
labels = np.load(labels_path)
print(data.shape)
print(labels.shape)

data_test = data[data_test_begin:data_test_end]
data_test_index = np.arange(0,data.shape[0])
data_train = data[np.delete(data_test_index,np.arange(data_test_begin,data_test_end))]

labels_test = labels[data_test_begin:data_test_end]
labels_test_index = np.arange(0,labels.shape[0])
labels_train = labels[np.delete(labels_test_index,np.arange(data_test_begin,data_test_end))]

print(data_train.shape)
print(labels_train.shape)

clf_sfnc_masked = linear_model.LogisticRegression(random_state=0, solver = 'lbfgs', max_iter = 1000)
clf_sfnc_masked.fit(data_train, labels_train)

s = clf_sfnc_masked.score(data_test,labels_test)

print(s)