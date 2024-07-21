import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TunedThresholdClassifierCV


df = pd.read_csv('csv/train.csv')
df2 = pd.read_csv('csv/test.csv')

X_full = df.drop(['Exited', 'id', 'CustomerId'], axis=1)
y = df['Exited']
X_test = df2.drop(['id', 'CustomerId'], axis=1)

object_cols = [i for i in X_full.columns if X_full[i].nunique() < 10 and X_full[i].dtype == 'object']
numeric_cols = [i for i in X_full.columns if X_full[i].dtype in ['float64', 'int64']]

my_cols = object_cols + numeric_cols
X = X_full[my_cols]

# for c in X.select_dtypes('object'):
#     X.loc[:, [c]], _ = X[c].factorize()

# discrete_feats = X.dtypes == int

# def make_mi_scores(X, y, discrete_feats):
#     mi_scores = mutual_info_regression(X, y, discrete_features=discrete_feats)
#     mi_scores = pd.Series(mi_scores, name='MI Scores', index= X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores

# mi_scores = make_mi_scores(X, y, discrete_feats)

# print(mi_scores)

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), object_cols)],
    remainder='passthrough')

model = CatBoostClassifier(n_estimators=200, 
                           learning_rate=0.1, 
                           verbose=0,
                           random_state=42,
                           leaf_estimation_iterations=10)


my_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

score = cross_val_score(my_pipe, X, y, cv=5, scoring='accuracy')

print(score)

# my_pipe.fit(X, y)

# preds = my_pipe.predict(X_test)

# X_test['Exited'] = preds

# subimission_final = pd.concat([df2['id'], X_test['Exited']], axis=1)

# subimission_final.to_csv('submission2.csv', index=False)