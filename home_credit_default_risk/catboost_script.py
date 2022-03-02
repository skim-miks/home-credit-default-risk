import pandas as pd
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

train = pd.read_csv('application_train.csv')
test = pd.read_csv('application_test.csv')

model = CatBoostClassifier()
model.load_model('/model_cat')

Y = train['TARGET']

cat_columns = [col for col in train.columns if train[col].dtype == 'object']

train[cat_columns] =train[cat_columns].astype(str)
test[cat_columns] =test[cat_columns].astype(str)

train['EXT_SOURCE_COMB'] = train['EXT_SOURCE_1'] + train['EXT_SOURCE_2']
train['EXT_SOURCE_COMB1'] = abs(train['EXT_SOURCE_1'] - train['EXT_SOURCE_2'])
train['EXT_SOURCE_COMB2'] = (train['EXT_SOURCE_1']+train['EXT_SOURCE_2'])/2

test['EXT_SOURCE_COMB'] = test['EXT_SOURCE_1'] + test['EXT_SOURCE_2']
test['EXT_SOURCE_COMB1'] = abs(test['EXT_SOURCE_1'] - test['EXT_SOURCE_2'])
test['EXT_SOURCE_COMB2'] = (test['EXT_SOURCE_1']+test['EXT_SOURCE_2'])/2

train = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)

k=5
kf = KFold(n_splits=k, random_state=42, shuffle = True)

model = CatBoostClassifier(
  iterations = 20000,
  depth=5,
  loss_function="Logloss",
  eval_metric="AUC",
  learning_rate=.02,
  random_seed=42,
  od_type='Iter', 
  od_wait=300,
  verbose=250
  )

y_valid_pred = 0 * Y
for idx, (train_index, valid_index) in enumerate(kf.split(train)):
  y_train, y_valid = Y.iloc[train_index], Y.iloc[valid_index]
  x_train, x_valid = train.iloc[train_index,:], train.iloc[valid_index,:]
  _train = Pool(x_train, label = y_train,cat_features=cat_columns)
  _valid = Pool(x_valid, label = y_valid,cat_features=cat_columns)
  print("\nFold ", idx)
  fit_model = model.fit(_train,
                        eval_set=_valid,
                        use_best_model = True,
                        plot=True
                        )
  pred = fit_model.predict_proba(x_valid)[:,1]
  print(" auc = ", roc_auc_score(y_valid, pred))
  y_valid_pred.iloc[valid_index] = pred
roc_auc_score(Y, y_valid_pred)

x_train, x_valid, y_train, y_valid = train_test_split(train, Y, test_size=.1, random_state=42)

model.fit(x_train, y_train, eval_set=(x_valid, y_valid), use_best_model = True, cat_features=cat_columns)

p = pd.DataFrame(model.predict_proba(test.iloc[:,1:])[:,1])
p = pd.concat([test.iloc[:,0].reset_index(drop=True),p.reset_index(drop=True)], axis=1)

p = p.rename(columns={0:'TARGET'})

p.to_csv('sub.csv', index=False)

model.save_model('model_cat')

import pickle

pickle.dump(model, open('cb_model.sav', 'wb'))

feat_list = list(train.columns)
pickle.dump(feat_list, open('feat_list.pkl', 'wb'))
