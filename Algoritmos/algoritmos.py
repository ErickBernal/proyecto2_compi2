import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB;



name_csv = "nac.csv"

def regLineal(column,dataPredic):
    data = pd.read_csv(name_csv)
   # print(data.info)
    x = np.asarray(data['Solola']).reshape(-1,1)#var independiente
    y = data['Ano']#var dependiente
    regr = linear_model.LinearRegression()
    regr.fit(x,y)

    y_pred = regr.predict(x)
    plt.title('Linear Regression')
    plt.scatter(x,y, color='black')
    plt.plot(x,y_pred, color='blue',linewidth=3)
    plt.ylim(0,20)
    print(regr.predict([[2030]]))
    plt.show()                   # Display the plotp


def regPolinomial(column,dataPredic):
    print("reg polinomial")
    dataset = pd.read_csv(name_csv)
    #X=dataset.iloc[:,1:2].values
    #y=dataset.iloc[:,2].values
    X = np.asarray(dataset[column]).reshape(-1,1)#var independiente
    y = dataset['NO']#var dependiente
    lin_reg=LinearRegression()
    lin_reg.fit(X,y)

    poly_reg=PolynomialFeatures(degree=4)
    X_poly=poly_reg.fit_transform(X)
    poly_reg.fit(X_poly,y)

    lin_reg2=LinearRegression()
    lin_reg2.fit(X_poly,y)

    X_grid=np.arange(min(X),max(X),0.1)
    X_grid=np.arange(min(X),max(X),0.1)
    X_grid=X_grid.reshape((len(X_grid),1))
    plt.scatter(X,y,color='red')
    plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
    plt.title('Polynomial Regression')
    plt.ylim(0,20)
    print(lin_reg2.predict( poly_reg.fit_transform([[dataPredic]]) ) )

    plt.show()


def clasificadorProcesosGaussiano():
    print('holi')


def bayes():
    le = preprocessing.LabelEncoder()
    df = pd.read_csv('tem1.csv')
    a = le.fit_transform(df.A)
    b = le.fit_transform(df.B)
    c = le.fit_transform(df.C)
    d = le.fit_transform(df.D)
    e = le.fit_transform(df.E)
    f = le.fit_transform(df.F)
    g = le.fit_transform(df.G)
    h = le.fit_transform(df.H)

    i = le.fit_transform(df.I)

    print('<conlumnas encoder, Bayes>')
    print('A: ',a)
    print('B: ',b)
    print('C: ',c)
    print('D: ',d)
    print('e: ',e)

    features = list(zip(a,b,c,d,e,f,g,h))
    print(features)

    model = GaussianNB()
    model.fit(features,i)
    predict = model.predict([[7,4,0,3,4,2,0,0]])
    print("Predict value: ", predict);

    #print('<Gauss>')
    #clf = DecisionTreeClassifier().fit(features,i)
    #tree.plot_tree(clf,filled=True)
   # plt.show()
    #calcular prediccion
    


def arbolDecision():
    print('arbol Desicion')
    df = pd.read_csv(name_csv)
    df = pd.get_dummies(data=df,drop_first=True)    
    Explicativas = df.drop(columns='E')#todas menos E
    objetivo = df.E
    print(objetivo)

    model = DecisionTreeClassifier()
    model.fit(X=Explicativas,y=objetivo)
    DecisionTreeClassifier()

    tree.plot_tree(decision_tree=model,feature_names=Explicativas.columns,filled=True)


def main():
    col = 'Solola'
    dataPredic = 50
    #bayes()
    regLineal(col, 50)
   # arbolDecision()
    return 0


if __name__ == "__main__":
    main()



#regLineal(col,dataPredic)
#regPolinomial(col,dataPredic)