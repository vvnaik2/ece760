import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kNN(x, X, k, Y):
    X = X - x
    X = X * X
    X = np.sum(X, axis=1)
    X = np.sqrt(X)
    knn_y = Y[np.argsort(X, axis=-1)[0:k]]
    return knn_y

d2z = pd.read_csv("D2z.txt", header=None, delimiter = ' ').to_numpy()

X = d2z[:, :-1]
Y = d2z[:, -1:]
colors_x = list()
colors_y = list()
colors_z = list()
for x in np.arange(-2, 2.1, 0.1):
    for y in np.arange(-2, 2.1, 0.1):
        prediction = kNN([x,y], X, 1, Y)
        colors_x.append(x)
        colors_y.append(y)
        colors_z.append(prediction[0][0])
colors_x = np.array(colors_x)
colors_y = np.array(colors_y)
colors_z = np.array(colors_z)
xin = X.T[0]
x2in = X.T[1]
y_out = Y.T[0]
plt.figure(figsize=(10, 10))
plt.scatter(colors_x[colors_z==0], colors_y[colors_z==0], marker='.', label='Test Result = Green o')
plt.scatter(colors_x[colors_z==1], colors_y[colors_z==1], marker='.', label='Test Result = Red +')
plt.scatter(xin[y_out==0], x2in[y_out==0], marker='o', label='Train set Green o')
plt.scatter(xin[y_out==1], x2in[y_out==1], marker='+', label='Train set Red +')
plt.legend(loc=4)
plt.savefig("P1_Plot.png")
plt.show()

# Spam Dataset
spam_df = pd.read_csv('emails.csv')
sdf_in = spam_df.drop(labels = ['Prediction', 'Email No.'], axis = 1).to_numpy()
sdf_out = pd.DataFrame(spam_df['Prediction']).to_numpy()
# Epsilon
epsilon = 1e-6

# Training set 1
X_train1 = sdf_in[1000:5000]
X_test1 = sdf_in[:1000]
Y_train1 = sdf_out[1000:5000]
Y_test1 = sdf_out[:1000]
# Training set 2
X_train2 = np.concatenate([sdf_in[:1000], sdf_in[2000:]])
X_test2 = sdf_in[1000:2000]
Y_train2 = np.concatenate([sdf_out[:1000], sdf_out[2000:]])
Y_test2 = sdf_out[1000:2000]
# Training set 3
X_train3 = np.concatenate([sdf_in[:2000], sdf_in[3000:]])
X_test3 = sdf_in[2000:3000]
Y_train3 = np.concatenate([sdf_out[:2000], sdf_out[3000:]])
Y_test3 = sdf_out[2000:3000]
# Training set 4
X_train4 = np.concatenate([sdf_in[:3000], sdf_in[4000:]])
X_test4 = sdf_in[3000:4000]
Y_train4 = np.concatenate([sdf_out[:3000], sdf_out[4000:]])
Y_test4 = sdf_out[3000:4000]
# Training set 5
X_train5 = sdf_in[:4000]
X_test5 = sdf_in[4000:]
Y_train5 = sdf_out[:4000]
Y_test5 = sdf_out[4000:]

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

Y_predict1 = predict(X_train1, Y_train1, X_test1, 1)
tp, tn, fp, fn, accuracy1, precision1, recall1 = compute_metrics(Y_predict1, Y_test1.T[0])
Y_predict2 = predict(X_train2, Y_train2, X_test2, 1)
tp, tn, fp, fn, accuracy2, precision2, recall2 = compute_metrics(Y_predict2, Y_test2.T[0])
Y_predict3 = predict(X_train3, Y_train3, X_test3, 1)
tp, tn, fp, fn, accuracy3, precision3, recall3 = compute_metrics(Y_predict3, Y_test3.T[0])
Y_predict4 = predict(X_train4, Y_train4, X_test4, 1)
tp, tn, fp, fn, accuracy4, precision4, recall4 = compute_metrics(Y_predict4, Y_test4.T[0])
Y_predict5 = predict(X_train5, Y_train5, X_test5, 1)
tp, tn, fp, fn, accuracy5, precision5, recall5 = compute_metrics(Y_predict5, Y_test5.T[0])
Q2_result = {
    "Fold": [1,2,3,4,5],
    "Accuracy": [accuracy1, accuracy2, accuracy3, accuracy4, accuracy5],
    "Precision": [precision1, precision2, precision3, precision4, precision5],
    "Recall": [recall1, recall3, recall3, recall4, recall5]
}
Q2_df = pd.DataFrame(Q2_result)
print(Q2_df)

K = [1,3,5,7,9]
#K = [10]

acc = []
accAvg = []
folds = [[X_train1, X_test1, Y_train1, Y_test1],
         [X_train2, X_test2, Y_train2, Y_test2],
         [X_train3, X_test3, Y_train3, Y_test3],
         [X_train4, X_test4, Y_train4, Y_test4],
         [X_train5, X_test5, Y_train5, Y_test5]]
for k in K:
    kAcc = []
    kAccAvg = 0
    for [X_tr, X_ts, Y_tr, Y_ts] in folds:
        Y_pr = predict(X_tr, Y_tr, X_ts, k)
        _, _, _, _, accuracy, _, _ = compute_metrics(Y_pr, Y_ts.T[0])
        kAcc.append(accuracy)
        kAccAvg += accuracy
    acc.append(kAcc)
    accAvg.append(kAccAvg/5)

acc1 = np.array(acc).T[0]
acc2 = np.array(acc).T[1]
acc3 = np.array(acc).T[2]
acc4 = np.array(acc).T[3]
acc5 = np.array(acc).T[4]

plt.figure(figsize=(5, 5))
plt.plot(K, accAvg, marker='.')
plt.xlabel("K")
plt.ylabel("Average Accuracy")
plt.title("Average Accuracy VS K")
plt.savefig("P4_AvgAccVsK.png")
plt.show()

P4_result = {
    "K": K,
    "Accuracy1": acc1,
    "Accuracy2": acc2,
    "Accuracy3": acc3,
    "Accuracy4": acc4,
    "Accuracy5": acc5,
}
P4_df = pd.DataFrame(P4_result)

print(P4_df)

Y_predict = predict(X_train1, Y_train1, X_test1, 5)
TPR = []
FPR = []
for c in np.arange(0.000, 1.001, 0.001):
    tp, tn, fp, fn, accuracy, precision, recall = compute_metrics(Y_predict, Y_test1.T[0], c)
    TPR.append(tp/(tp+fn))
    FPR.append(fp/(fp+tn))
plt.figure(figsize=(5, 5))
plt.plot(FPR, TPR, marker='.')
plt.show()
