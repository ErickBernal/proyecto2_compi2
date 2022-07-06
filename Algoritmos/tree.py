import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd

data = pd.read_csv("pred.csv")

#modo atunomo 
arr =[]
cont = 0
for x in data:
    if cont!=5 and cont != 0:
        arr.append(data.iloc[:,cont].values)
    
    cont = cont + 1
    

#paso a paso
A = data.iloc[:, 1].values
B = data.iloc[:, 2].values
C = data.iloc[:, 3].values
D = data.iloc[:, 4].values
P = data.iloc[:, 5].values

print(A)
print(B)
print(C)
print(D)
print(P)
#features = list(zip(A,B,C,D))

features = list(zip(*arr))
print(features)

clf = DecisionTreeClassifier().fit(features, P)
plot_tree(clf, filled=True)
plt.show()
plt.savefig('pred.png', bbox_inches='tight')
