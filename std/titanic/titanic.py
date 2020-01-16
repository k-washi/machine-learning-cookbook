import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')

#前処理
train_x = train.drop(['Survived'], axis=1)
train_y = train['Survived']

test_x = test.copy()

train_x = train_x.drop(['PassengerId'], axis=1)
test_x = test_x.drop(['PassengerId'], axis=1)

train_x = train_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)
test_x = test_x.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#print(train_x.head())
'''
   Pclass     Sex   Age  SibSp  Parch     Fare Embarked
0       3    male  22.0      1      0   7.2500        S
1       1  female  38.0      1      0  71.2833        C
2       3  female  26.0      0      0   7.9250        S
3       1  female  35.0      1      0  53.1000        S
4       3    male  35.0      0      0   8.0500        S
'''

#カテゴリ変数を数字に変更
for c in ['Sex', 'Embarked']:
  le = LabelEncoder()
  le.fit(train_x[c].fillna('NA'))

  train_x[c] = le.transform(train_x[c].fillna('NA'))
  test_x[c] = le.transform(test_x[c].fillna('NA'))

#print(train_x.head())
'''
   Pclass  Sex   Age  SibSp  Parch     Fare  Embarked
0       3    1  22.0      1      0   7.2500         3
1       1    0  38.0      1      0  71.2833         0
2       3    0  26.0      0      0   7.9250         3
3       1    0  35.0      1      0  53.1000         3
4       3    1  35.0      0      0   8.0500         3
'''

model = XGBClassifier(n_estimators=20, random_state=71)
model.fit(train_x, train_y)

#予測確率(ラベル1)を出力
pred = model.predict_proba(test_x)[:, 1]

#予測確率を2値に変換
pred_label = np.where(pred > 1, 0, 1)

#Kaggle 提出用に変換
sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pred_label})
sub.to_csv('titanic_submission.csv', index=False)

# クロスバリデーション----------------------------------------------------------------------------------
#Grid Search モデルのチューニング 
param_space = {
   'max_depth': [3, 5, 7],
   'min_child_weight': [1., 2., 4.]
}

param_comb = itertools.product(param_space['max_depth'], param_space['min_child_weight'])

params = []
scores = []

for max_depth, min_child_weight in param_comb:
   #accurracyは小さな改善を捉えづらいため、loglossも用いる。loglossは予測確率が外れているほどペナルティが大きい
   scores_accuracy = []
   scores_logloss = []
   # 学習データ4分割, 
   kf = KFold(n_splits=4, shuffle=True, random_state=71)
   for tr_idx, va_idx, in kf.split(train_x):
      #学習データ => 学習データ, バリデーションデータ
      #https://note.nkmk.me/python-pandas-at-iat-loc-iloc/
      tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
      tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
 
      model = XGBClassifier(n_estimators=20, random_state=71, max_depth=max_depth, min_child_weight=min_child_weight)
      model.fit(tr_x, tr_y)

      va_pred = model.predict_proba(va_x)[:, 1]

      logloss = log_loss(va_y, va_pred)
      accuracy = accuracy_score(va_y, va_pred > 0.5)

      scores_logloss.append(logloss)
      scores_accuracy.append(accuracy)

   logloss = np.mean(scores_logloss)
   accuracy = np.mean(scores_accuracy)

   #print(f"logloss: {logloss:.4f}, accuracy: {accuracy:.4f}")

   scores.append(logloss)
   params.append((max_depth, min_child_weight))

#Grid Searchで最もスコアが良いものをベストパラメータとする。

best_idx = np.argsort(scores)[0]
best_param = params[best_idx]
best_score = scores[best_idx]
print(f'max_depth: {best_param[0]}, min_child_w: {best_param[1]}, logloss: {best_score}')
#max_depth: 5, min_child_w: 2.0, logloss: 0.42250640765538333
