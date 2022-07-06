import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("pred.csv")

A = df.iloc[:, 1].values
B = df.iloc[:, 2].values
C = df.iloc[:, 3].values
D = df.iloc[:, 4].values
P = df.iloc[:, 5].values

f = df['A']



arr=[]
for a in df:
    print(a)
    arr.append(a[0])




print("\n",A)
print(B)
print(C)
print(D)
print(P)

features = list(zip(A, B, C, D))

print("\nfeatures:\n",features)

model = GaussianNB()
model.fit(features, P)

predict = model.predict([[5, 500, 200, False]])
print("Predict value: ", predict)
plt.scatter(B, P)
plt.show()

