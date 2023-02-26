from logging import logMultiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

sf_data  = pd.read_csv('data/emails.csv')
sf_df = pd.DataFrame(sf_data)
sf_feature_data = sf_df.drop(["Prediction","Email No."], axis='columns').to_numpy()
sf_label_data = sf_df[['Prediction']].to_numpy()

epsilon = 0.00000001

def mean_and_std(X):
    return np.mean(X, 0, keepdims=True), np.std(X, 0, keepdims=True)
def normalize(X):
    mean, std = mean_and_std(X)
    X_normal = (X - mean)/(std+epsilon)
    return X_normal

sf_feature_data = normalize(sf_feature_data)

X_train1 = sf_feature_data[1000:5000]
X_test1 = sf_feature_data[:1000]
Y_train1 = sf_label_data[1000:5000]
Y_test1 = sf_label_data[:1000]

X_train2 = np.concatenate([sf_feature_data[:1000],sf_feature_data[2000:]])
X_test2 = sf_feature_data[1000:2000]
Y_train2 = np.concatenate([sf_label_data[:1000],sf_label_data[2000:]])
Y_test2 = sf_label_data[1000:2000]

X_train3 = np.concatenate([sf_feature_data[:2000],sf_feature_data[3000:]])
X_test3 = sf_feature_data[2000:3000]
Y_train3 = np.concatenate([sf_label_data[:2000],sf_label_data[3000:]])
Y_test3 = sf_label_data[2000:3000]

X_train4 = np.concatenate([sf_feature_data[:3000],sf_feature_data[4000:]])
X_test4 = sf_feature_data[3000:4000]
Y_train4 = np.concatenate([sf_label_data[:3000],sf_label_data[4000:]])
Y_test4 = sf_label_data[3000:4000]

X_train5 = sf_feature_data[:4000]
X_test5 = sf_feature_data[4000:]
Y_train5 = sf_label_data[:4000]
Y_test5 = sf_label_data[4000:]

#print(X_train1)

num_of_iter = 1000
lr = 0.01


def sigmoid(x):
  return 1/(1+np.exp(-x))

def train_model(X,Y,lr,num_of_iter): 
  W = np.zeros(X.shape[1])
  #W = np.zeros((1, X.shape[1]))
  print(W)
  print(W.shape[0])
  b = 0
  n = X.shape[0]
  for i in range(num_of_iter):
    print("Iteration number : {}".format(i))
    # Calculate Y_hat with the existing weights
    Y_hat = sigmoid(np.dot(W, X.T)+b)
    #print(Y_hat)
    # Calculate Loss
    loss = (-1/n)*np.sum(Y.T*np.log(Y_hat+epsilon)+(1-Y.T)*np.log(1-Y_hat+epsilon))
    print(loss)
    # Calculate gradient
    dw = (1/n)*(np.dot(X.T, (Y_hat-Y.T).T))
    db = (1/n)*(np.sum((Y_hat-Y.T).T))
    # Update the weights
    W = W - lr*dw.T
    b = b - lr*db
  #print(W)
  return W , b

W1, b1 = train_model(X_train1, Y_train1, lr, num_of_iter)
W2, b2 = train_model(X_train2, Y_train2, lr, num_of_iter)
W3, b3 = train_model(X_train3, Y_train3, lr, num_of_iter)
W4, b4 = train_model(X_train4, Y_train4, lr, num_of_iter)
W5, b5 = train_model(X_train5, Y_train5, lr, num_of_iter)

def test_model(W,b,X,Y,c=0.5):
  Y_hat = sigmoid(np.dot(W, X.T)+b)
  Y_hat = (Y_hat >= c) + 0
  tp = np.sum(np.logical_and(Y==1, Y_hat.T==1))
  fp = np.sum(np.logical_and(Y==0, Y_hat.T==1))
  tn = np.sum(np.logical_and(Y==0, Y_hat.T==0))
  fn = np.sum(np.logical_and(Y==1, Y_hat.T==0))
  accuracy = np.sum(Y_hat.T == Y)/len(Y)
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  return tp, tn, fp, fn, accuracy, precision, recall

tp , tn , fp , fn , accuracy1, precision1, recall1 = test_model(W1,b1,X_test1,Y_test1)
tp , tn , fp , fn , accuracy2, precision2, recall2 = test_model(W2,b2,X_test2,Y_test2)
tp , tn , fp , fn , accuracy3, precision3, recall3 = test_model(W3,b3,X_test3,Y_test3)
tp , tn , fp , fn , accuracy4, precision4, recall4 = test_model(W4,b4,X_test4,Y_test4)
tp , tn , fp , fn , accuracy5, precision5, recall5 = test_model(W5,b5,X_test5,Y_test5)

print("Testing Complete")
print("Learning rate        : {}".format(lr))
print("Number of iterations : {}".format(num_of_iter))
print("Fold 1 accuracy  : {}".format(accuracy1))
print("Fold 1 precision : {}".format(precision1))
print("Fold 1 recall    : {}".format(recall1))
print("Fold 2 accuracy  : {}".format(accuracy2))
print("Fold 2 precision : {}".format(precision2))
print("Fold 2 recall    : {}".format(recall2))
print("Fold 3 accuracy  : {}".format(accuracy3))
print("Fold 3 precision : {}".format(precision3))
print("Fold 3 recall    : {}".format(recall3))
print("Fold 4 accuracy  : {}".format(accuracy4))
print("Fold 4 precision : {}".format(precision4))
print("Fold 4 recall    : {}".format(recall4))
print("Fold 5 accuracy  : {}".format(accuracy5))
print("Fold 5 precision : {}".format(precision5))
print("Fold 5 recall    : {}".format(recall5))

w, b = train_model(X_train1, Y_train1, lr, num_of_iter)

Y_hat = sigmoid(np.dot(w, X_test1.T)+b)

TPR_lr = []
FPR_lr = []
for c in np.arange(0.00, 1.01, 0.01):
  tp , tn , fp , fn , accuracy, precision, recall = test_model(w,b,X_test1,Y_test1,c)
  print("Values {} {} {} {}".format(tp,tn,fp,fn))
  TPR_lr.append(tp/(tp+fn))
  FPR_lr.append(fp/(fp+tn))
plt.figure(figsize=(5, 5))
plt.plot(FPR_lr, TPR_lr, marker='.')
plt.show()

# Adding KNN here
 
def kNN(x, X, k, Y):
    X = X - x
    X = X * X
    X = np.sum(X, axis=1)
    X = np.sqrt(X)
    knn_y = Y[np.argsort(X, axis=-1)[0:k]]
    return knn_y
def compute_metrics(Y_predict, Y, c = 0.5):
    Y_predict = (Y_predict >= c) + 0
    tp = np.sum(np.logical_and(Y==1, Y_predict.T==1))
    fp = np.sum(np.logical_and(Y==0, Y_predict.T==1))
    tn = np.sum(np.logical_and(Y==0, Y_predict.T==0))
    fn = np.sum(np.logical_and(Y==1, Y_predict.T==0))
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return tp, tn, fp, fn, accuracy, precision, recall
def predict(X, Y, X_test, k):
    y_test = []
    for x in X_test:
        knn = kNN(x, X, k, Y)
        y_test.append(np.sum(knn)/k)
    return np.array(y_test)
Y_predict = predict(X_train1, Y_train1, X_test1, 5)
TPR_knn = []
FPR_knn = []
for c in np.arange(0.00, 1.01, 0.01):
    tp, tn, fp, fn, accuracy, precision, recall = compute_metrics(Y_predict, Y_test1.T[0], c)
    TPR_knn.append(tp/(tp+fn))
    FPR_knn.append(fp/(fp+tn))
TPR_knn.append(0)
FPR_knn.append(0)


plt.figure(figsize=(5, 5))
plt.plot(FPR_knn, TPR_knn, marker='.')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(FPR_lr, TPR_lr, marker='.', label='Logistic Regression')
plt.plot(FPR_knn, TPR_knn, marker='.', label='KNN ROC')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC")
plt.legend()
plt.savefig("P5_ROC_LR_VS_KNN.png")
plt.show()
