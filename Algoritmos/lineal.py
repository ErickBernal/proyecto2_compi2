import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;
from sklearn.linear_model import LinearRegression;
from sklearn.metrics import mean_squared_error, r2_score;


data = pd.read_csv("nac.csv");

X = np.asarray(data['Ano']).reshape(-1, 1);
Y = data['Republica'];

linear_regression = LinearRegression();
linear_regression.fit(X, Y);
Y_pred = linear_regression.predict(X);

print(Y_pred);
print("Error medio: ", mean_squared_error(Y, Y_pred, squared=True));
print("Coef: ", linear_regression.coef_);
print("R2: ", r2_score(Y, Y_pred));

plt.scatter(X, Y);
plt.plot(X, Y_pred, color='red');
plt.show();

Y_new = linear_regression.predict([[2025]]);
print(Y_new);

