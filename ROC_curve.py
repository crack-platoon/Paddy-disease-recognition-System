import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import random
from random import shuffle

data = pd.read_csv("F:/Projects/hhh/dataset/train/train.csv", index_col=0)

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

y = label_binarize(target, classes=[1, 2, 3])

n_classes = y.shape[1]

random_state = np.random.RandomState(0)

    #model evaluation train test split
X_train, X_test, y_train, y_test = train_test_split(data,y, test_size = 0.30, random_state = 0,shuffle=True)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

print y_score[:,2]
print y
print n_classes

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



print fpr
print tpr

print roc_auc



lw=2
# Plot all ROC curves
plt.figure()

colors = cycle(['red', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.legend(loc="lower right")
plt.show()


