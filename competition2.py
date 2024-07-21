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


df = pd.read_csv('csv/train.csv')

X_full = df.drop('Exited', axis=1)
y = df['Exited']

object_cols = [i for i in X_full.columns if X_full[i].nunique() < 10 and X_full[i].dtype == 'object']
numeric_cols = [i for i in X_full.columns if X_full[i].dtype in ['float64', 'int64']]

my_cols = object_cols + numeric_cols
X = X_full[my_cols]


object_transform = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

numeric_transform = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, numeric_cols),
        ('cat', object_transform, object_cols)
    ])

model = CatBoostClassifier(n_estimators=200, learning_rate=0.1, verbose=0)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

my_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

my_pipe.fit(X_train, y_train)

preds = my_pipe.predict(X_valid)

print(accuracy_score(y_valid, preds))