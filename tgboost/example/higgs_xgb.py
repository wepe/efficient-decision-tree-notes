import xgboost as xgb
import pandas as pd
import numpy as np

train = pd.read_csv('../../data/train.csv').drop(["EventId", "Weight"], axis=1)
val = pd.read_csv('../../data/test.csv').drop(["EventId", "Weight"], axis=1)
train.replace(to_replace=-999., value=np.nan, inplace=True)
train.replace(to_replace='s', value=1, inplace=True)
train.replace(to_replace='b', value=0, inplace=True)
val.replace(to_replace=-999, value=np.nan, inplace=True)
val.replace(to_replace='s', value=1, inplace=True)
val.replace(to_replace='b', value=0, inplace=True)


train_y = train.Label
train_X = train.drop('Label', axis=1)
val_y = val.Label
val_X = val.drop('Label', axis=1)

print train_y.shape, train_y.sum()
print val_y.shape, val_y.sum()

dtrain = xgb.DMatrix(train_X, label=train_y)
dval = xgb.DMatrix(val_X, label=val_y)

params = {'booster':'gbtree',
          'objective': 'binary:logistic',
          'eta': 0.3,
          'max_depth': 7,
          'num_boost_round': 10000,
          'scale_pos_weight': 1.0,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'colsample_bylevel': 1.0,
          'min_sample_split': 50,
          'min_child_weight': 1,
          'lambda': 1,
          'gamma': 0.01,
          'eval_metric': "auc",
          'maximize': False,
          'num_thread': 16}

watchlist = [(dval, 'val')]
model = xgb.train(params, dtrain,num_boost_round=10000, early_stopping_rounds=20, evals=watchlist)

"""
Stopping. Best iteration:
[35]	val-auc:0.90931
"""