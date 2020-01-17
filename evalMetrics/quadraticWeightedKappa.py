from sklearn.metrics import confusion_matrix, cohen_kappa_score

def quadratic_weighted_kappa(c_matrix):
  num = 0.
  denom = 0.

  for i in range(c_matrix.shape[0]):
    for j in range(c_matrix.shape[1]):
      n = c_matrix.shape[0]
      wij = ((i-j) ** 2.)
      oji = c_matrix[i,j]
      eij = c_matrix[i, :].sum() * c_matrix[:,j].sum() / c_matrix.sum()

      num += wij * oji
      denom += wij * eij

  return 1. - num / denom


y_true = [1,2,3,4,3]
y_pred = [2,2,4,4,5]

c_matrix = confusion_matrix(y_true, y_pred, labels=[1,2,3,4,5])

kappa = quadratic_weighted_kappa(c_matrix)
#print(kappa)
#0.6153846153846154

kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
#print(kappa)
#0.6153846153846154