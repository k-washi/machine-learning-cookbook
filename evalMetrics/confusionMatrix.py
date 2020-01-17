from sklearn.metrics import confusion_matrix
import numpy as np

y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

#TP: 予測値が正かつ、予測が正しい
tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
#TN: 予測値が負かつ、予測が正しい
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
#FP: 予測値が正かつ、予測が誤り
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
#FN: 予測値が負かつ、予測が誤り
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp],[fn, tn]])
#print(confusion_matrix1)
#[[3 1]
# [2 2]]

confusion_matrix2 = confusion_matrix(y_true, y_pred)
#print(confusion_matrix2)
#[[2 1]
# [2 3]]