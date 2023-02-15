import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

# CONSTANTS
FEATURE_1 = 'X1'
FEATURE_2 = 'X2'
CLASS_LABEL_COLUMN_NAME = 'Y'
COLUMN_NAMES = [FEATURE_1, FEATURE_2, CLASS_LABEL_COLUMN_NAME] # Assumption is all text files have only 2 features 

def read_csv(file_name, column_names):
    return pd.read_csv(file_name, sep=' ', header=None, names=column_names)

Dbig_dataset = read_csv(file_name = 'Dbig.txt', column_names = COLUMN_NAMES)
Dbig_permutation = Dbig_dataset.iloc[np.random.RandomState(seed = 5).permutation(len(Dbig_dataset))].reset_index(drop=True)
Dbig_train = Dbig_permutation[0:8192]
Dbig_test = Dbig_permutation[8192:]
D32 = Dbig_train[0:32]
D32_X = D32.loc[:, D32.columns != 'Y']
D32_Y = D32.loc[:, D32.columns == 'Y']

D128 = Dbig_train[:128]
D128_X = D128.loc[:, D128.columns != 'Y']
D128_Y = D128.loc[:, D128.columns == 'Y']

D512 = Dbig_train[:512]
D512_X = D512.loc[:, D512.columns != 'Y']
D512_Y = D512.loc[:, D512.columns == 'Y']

D2048 = Dbig_train[:2048]
D2048_X = D2048.loc[:, D2048.columns != 'Y']
D2048_Y = D2048.loc[:, D2048.columns == 'Y']

D8192 = Dbig_train[:]
D8192_X = D8192.loc[:, D8192.columns != 'Y']
D8192_Y = D8192.loc[:, D8192.columns == 'Y']

test_X = Dbig_test.loc[:, Dbig_test.columns != 'Y']
test_Y = Dbig_test.loc[:, Dbig_test.columns == 'Y']

D32_d3 = DecisionTreeClassifier(criterion = "entropy").fit(D32_X, D32_Y)
D32_pred = D32_d3.predict(test_X)
D32_accuracy = metrics.accuracy_score(test_Y, D32_pred)
n32 = D32_d3.tree_.node_count

D128_d3 = DecisionTreeClassifier(criterion = "entropy").fit(D128_X, D128_Y)
D128_pred = D128_d3.predict(test_X)
D128_accuracy = metrics.accuracy_score(test_Y, D128_pred)
n128 = D128_d3.tree_.node_count

D512_d3 = DecisionTreeClassifier(criterion = "entropy").fit(D512_X, D512_Y)
D512_pred = D512_d3.predict(test_X)
D512_accuracy = metrics.accuracy_score(test_Y, D512_pred)
n512 = D512_d3.tree_.node_count

D2048_d3 = DecisionTreeClassifier(criterion = "entropy").fit(D2048_X, D2048_Y)
D2048_pred = D2048_d3.predict(test_X)
D2048_accuracy = metrics.accuracy_score(test_Y, D2048_pred)
n2048 = D2048_d3.tree_.node_count

D8192_d3 = DecisionTreeClassifier(criterion = "entropy").fit(D8192_X, D8192_Y)
D8192_pred = D8192_d3.predict(test_X)
D8192_accuracy = metrics.accuracy_score(test_Y, D8192_pred)
n8192 = D8192_d3.tree_.node_count

n = [n32, n128, n512, n2048, n8192]
acc = [D32_accuracy, D128_accuracy, D512_accuracy, D2048_accuracy, D8192_accuracy]
err = [1-D32_accuracy, 1-D128_accuracy, 1-D512_accuracy, 1-D2048_accuracy, 1-D8192_accuracy]
print("Printing n and errn lists respectively:")
print(n)
print(err)
plt.plot(n, err) 
