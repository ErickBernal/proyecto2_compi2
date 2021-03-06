from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
import pickle

#cargar los datos en dataset
iris = datasets.load_iris()

X = iris.data
y =iris.target


#separar los datos de entranamiento y prueba
x_train,x_test,y_train,y_test = train_test_split(X,y)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svc_model = SVC()

#entrenar modelos
lin_reg_fit = lin_reg.fit(x_train, y_train)
log_reg_fit = log_reg.fit(x_train, y_train)
svc_model_fit = svc_model.fit(x_train, y_train)

#guardar datos en archivo vicodi 

with open('lin_reg.pkl','wb' ) as li:
    pickle.dump(lin_reg_fit, li)

with open('log_reg.pkl','wb') as lo:
    pickle.dump(log_reg_fit, lo)

with open('svc_model.pkl' , 'wb') as sv:
    pickle.dump(svc_model_fit, sv)


