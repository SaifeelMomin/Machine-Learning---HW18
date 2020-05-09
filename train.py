import pandas as pd
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import KFold

from joblib import dump 
from preprocess import prep_data


df = pd.read_csv("fish_participant.csv")
df = df.assign(lavg = (df["Length1"] + df["Length2"] + df["Length3"])/3)
X, y = prep_data(df)
# X = df[['Length1', 'Length2', 'Length3', 'Width', 'Height']].values
# y = df['Weight']

kf = KFold(n_splits=10, shuffle=True, random_state=444)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


## rf ## 
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred)


## extratr ## 
extra_tree = ExtraTreeRegressor()
reg = BaggingRegressor(extra_tree).fit(X_train, y_train)
p_reg = reg.predict(X_test)

mse = mean_squared_error(y_test, p_reg)

joblib.dump(reg, "reg.joblib")

## adaboost reg ##
ada = AdaBoostRegressor(base_estimator = reg, loss = 'exponential')
ada.fit(X_train, y_train)
p_ada = ada.predict(X_test)

mse = mean_squared_error(y_test, p_ada)


## gradientboost reg ##
gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
p_gbr = gbr.predict(X_test)

mse = mean_squared_error(y_test, p_gbr)
