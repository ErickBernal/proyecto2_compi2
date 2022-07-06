import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("pred.csv")

X = data.iloc[:, 0].values.reshape(-1, 1)
Y = data.iloc[:, 2].values.reshape(-1, 1)# [:, x], x indica la posicionde la columna a analizar 

poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
Y = poly.fit_transform(Y)

linear_regression = LinearRegression()
linear_regression.fit(X, Y)
Y_pred = linear_regression.predict(X)

print(Y_pred)
print("Error medio: ", mean_squared_error(Y, Y_pred, squared=False))
print("R2: ", r2_score(Y, Y_pred))

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

Y_new = linear_regression.predict(poly.fit_transform([[50]]))
print("Predicción:",Y_new)