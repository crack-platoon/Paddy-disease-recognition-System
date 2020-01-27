import cv2
import numpy as np
import random
from random import shuffle
import math
import scipy as sp
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support as score


file = open("file_name.txt","r")
files = file.read()
image = cv2.imread(files)
data = pd.read_csv("F:/Projects/hhh/dataset/Backup/train.csv", index_col=0)

#data = data.sample(frac=1)
#print data

data.fillna(0,inplace = True)

    #SVM initialization

cols = [col for col in data.columns if col not in ['type']]

train_data = data[cols]

#print train_data

train_data.fillna(0)

#print train_data

target = data['type']

#print target


    #model evaluation train test split
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 0,shuffle=True)

clf = svm.SVC(kernel='linear')

clf_acc = svm.SVC(kernel='linear')

clf.fit(train_data,target)

clf_acc.fit(data_train,target_train)

print data_train
print data_test
