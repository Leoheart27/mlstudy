import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv('csv/train.csv')

X_train_full = df.drop('Exited', axis=1)
y_target = df['Exited']

categorical_cols = [i for i in X_train_full.columns if X_train_full[i].nunique() < 10 and X_train_full[i].dtype == 'object']
cardinal_cols = [i for i in X_train_full.columns if X_train_full[i].dtype in ['float64', 'int64']]
valid_cardinal_col = ['CreditScore', 
                      'Age', 'Tenure', 
                      'Balance', 'NumOfProducts', 
                      'HasCrCard', 'IsActiveMember', 
                      'EstimatedSalary']

best_valid_cols = categorical_cols + valid_cardinal_col
X_train_best = X_train_full[best_valid_cols]


X_train, X_valid, y_train, y_valid = train_test_split(X_train_best, y_target, test_size=0.2)

cat_cols = [i for i in X_train.columns if X_train[i].dtype == 'object']
num_cols = [i for i in X_train.columns if X_train[i].dtype in ['float64', 'int64']]

my_cols = cat_cols + num_cols
X_train_final = X_train[my_cols].copy()
X_valid_final = X_valid[my_cols].copy()

categorical_transform = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
numerical_transform = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transform, num_cols),
        ('cat', categorical_transform, cat_cols)
    ])

model = CatBoostRegressor(n_estimators=200, learning_rate=0.1)

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

my_pipeline.fit(X_train_final, y_train)

preds = my_pipeline.predict(X_valid_final)

print(mean_absolute_error(y_valid, preds))