import streamlit as st
import pickle
import pandas as pd
from PIL import Image

#extraer los archivos pickle
with open('lin_reg.pkl','rb') as li:
    lin_reg = pickle.load(li)

with open('log_reg.pkl','rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_model.pkl','rb') as sv:
    svc_model = pickle.load(sv)



#-----------------------------------------------------MODELOS-----------------------------------------------------------------------
def convertirStrig_to_arr_obj(cadena):
    print('')
    cadena = cadena.strip()
    arr = cadena.split(",")
    ret =[]
    for a in arr:
        b = a.strip()#quitamos los espacios al inicio y al final
        if b.isdigit():
            b = int(b)
        else:
            b = float(b)
        ret.append(b)
    return ret



def algoritmo_lineal(name_csv,data_x,data_y,data_predic):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    path = './data/'+ name_csv
    data = pd.read_csv(path)

    X = np.asarray(data[data_x]).reshape(-1, 1)
    Y = data[data_y]

    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)

    #print(Y_pred)
   # print("Error medio: ", mean_squared_error(Y, Y_pred, squared=True))
   # print("Coef: ", linear_regression.coef_)
   # print("R2: ", r2_score(Y, Y_pred))

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

    Y_new = linear_regression.predict([[data_predic]])
    #print(Y_new)
    plt.savefig(path+".png", bbox_inches='tight')
    return Y_new


def algorito_redes_neuronales(name_csv,name_col_answ,arr_to_evaluate):
    from multiprocessing.connection import answer_challenge
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier

    # 1
    path = './data/'+ name_csv

    df = pd.read_csv(path)
    le = preprocessing.LabelEncoder()

    labelList=[]
    for name_col in df:
        if name_col_answ!=name_col:
            labelList.append(name_col)

    #labelList       =   ['A','B','C','D']
    #answ            =   'E'
    answ            =   name_col_answ
    #    toPredict       =   [[12.3,33,250,True]]
    toPredict = [convertirStrig_to_arr_obj(arr_to_evaluate)]
    ##### Desde 3

    L = pd.DataFrame()
    firstRun = True
    for x in labelList:
        pivot = pd.DataFrame(df[x])
        if (firstRun):
            L = pivot
            firstRun = False
        else:
            L = pd.concat([L, pivot.reindex(L.index)], axis=1)

    L_answ = pd.DataFrame(df[answ])
    x_train, x_test, y_train, y_test = train_test_split(L, L_answ, test_size=0.5, random_state=13)

    #------------------------------------------------------------------------------------------------------------------------------------
    ### 4

    scaler  =   StandardScaler()
    scaler.fit(x_train)

    x_train =   scaler.transform(x_train)
    x_test  =   scaler.transform(x_test)

    #------------------------------------------------------------------------------------------------------------------------------------
    ### 5

    MLP     =   MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=1000)
    MLP.fit(x_train, y_train.values.ravel())

    #------------------------------------------------------------------------------------------------------------------------------------
    ### 6

    predictions = MLP.predict(x_test)
    arr=[]
    print("Predictions: \n",predictions)
    arr.append(predictions)
    prediction = MLP.predict(toPredict)
    print("Prediction: ", prediction)
    arr.append(prediction)

    #------------------------------------------------------------------------------------------------------------------------------------
    ### 7

    print(confusion_matrix(y_test, predictions))
    arr.append(confusion_matrix(y_test, predictions))
    print(classification_report(y_test,predictions))
    arr.append(classification_report(y_test,predictions))
    return arr


def algoritmo_tree(name_csv,col_interes):
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    import pandas as pd

    path ="./data/"+name_csv
    data = pd.read_csv(path)


    arr =[]
    cont = 0
    P = data.iloc[:,0].values
    for x in data:
        if col_interes != x:
            arr.append(data.iloc[:,cont].values)
        elif x == col_interes:
            P = data.iloc[:,cont].values
        cont = cont + 1
  
    features = list(zip(*arr))

    clf = DecisionTreeClassifier().fit(features, P)
    plot_tree(clf, filled=True)
    path ="./data/"+name_csv+".png"
    plt.savefig(path, bbox_inches='tight')
    #plt.show()


def algoritmo_Gauss(name_csv,col_base,col_interes,arr_to_evaluate):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.naive_bayes import GaussianNB

    path ="./data/"+name_csv
    data = pd.read_csv(path)


    arr =[]
    cont = 0
    P = data.iloc[:,0].values
    for x in data:
        if col_interes != x:
            arr.append(data.iloc[:,cont].values)
        elif x == col_interes:
            P = data.iloc[:,cont].values
        cont = cont + 1
  
    features = list(zip(*arr))

    model = GaussianNB()
    model.fit(features, P)
    toPredict = [convertirStrig_to_arr_obj(arr_to_evaluate)]
    predict = model.predict(toPredict)
    plt.scatter(col_base,col_interes)

    path ="./data/"+name_csv+".png"
    plt.savefig(path, bbox_inches='tight')
    return predict

def algorito_polinomial(name_csv,pos_X_col,pos_y_col,predict_):
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    path = './data/' + name_csv
    data = pd.read_csv(path)

    X = data.iloc[:, pos_X_col].values.reshape(-1, 1)
    Y = data.iloc[:, pos_y_col].values.reshape(-1, 1)# [:, x], x indica la posicionde la columna a analizar 

    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)
    Y = poly.fit_transform(Y)

    linear_regression = LinearRegression()
    linear_regression.fit(X, Y)
    Y_pred = linear_regression.predict(X)

    #print(Y_pred)
    #print("Error medio: ", mean_squared_error(Y, Y_pred, squared=False))
    #print("R2: ", r2_score(Y, Y_pred))

    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
   # plt.show()

    Y_new = linear_regression.predict(poly.fit_transform([[predict_]]))
    #print("Predicción:",Y_new)
    path ="./data/"+name_csv+".png"
    plt.savefig(path, bbox_inches='tight')

    return Y_new#la prediccion

#----------------------------------------------------------------------------------------------------------------------------
def main():
    name_csv = './data/nac.csv'
    #titulo
    st.title('Proyecto 2 / Compiladores 2, Erick_Bernal')
    #titulo de side bar
    st.sidebar.header('Seleccion de parametros')
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(dataframe)

    #escojer el modelo
    option = ('Linear Regression' , 'Redes Neuronales','Arbol','Gauss','Regresion Polinomial')
    model = st.sidebar.selectbox('Seleccione el modelo a utilizar', option)

    Ingrese_x = ''
    Ingrese_y = ''
    Ingrese_z = ''
    if model == 'Linear Regression':
        Ingrese_x = st.sidebar.text_input('Ingrese el valor de X', 'Ano')
        Ingrese_y = st.sidebar.text_input('Ingrese el valor de y', 'Republica')
        ingrese_prediccionLineal = st.sidebar.text_input('Ingrese valor a predecir', '2025')
    elif model == 'Redes Neuronales':
        Ingrese_x = st.sidebar.text_input('Nombre de Columna objetivo', 'E')
        Ingrese_y = st.sidebar.text_input('Ingrese el vector a evaluar (separador por coma)','12.3,33,250,1')
    elif model == 'Arbol':
        Ingrese_x = st.sidebar.text_input('Nombre de Columna objetivo', 'E')
    elif model =='Gauss':
        Ingrese_x = st.sidebar.text_input('nombre, columna Base', 'B')
        Ingrese_y = st.sidebar.text_input('Nombre de Columna objetivo', 'E')
        Ingrese_z = st.sidebar.text_input('Ingrese el vector a evaluar (separador por coma)','5, 500, 200, 0')
    elif model =='Regresion Polinomial':
        Ingrese_x = st.sidebar.text_input('Ingrese Id(#) de la columna de interes para X', '0')
        Ingrese_y = st.sidebar.text_input('Ingrese Id(#) de la columna de interes para Y', '2')
        Ingrese_z = st.sidebar.text_input('Dato a predecir', '50')



        #st.write('x =', ingrese_prediccionLineal)


    if st.sidebar.button('Compilar'):
        if model == 'Linear Regression':
            st.subheader('<Regresion Lineal>')
            if uploaded_file is not None:
                pred =  algoritmo_lineal(uploaded_file.name, Ingrese_x, Ingrese_y, int(ingrese_prediccionLineal))
                st.text('Prediccion: ')
                st.success(pred)

                path = './data/' + uploaded_file.name + '.png'
                image = Image.open(path)
                st.image(image,caption='Prediccion lineal')

        elif model == 'Redes Neuronales':
            st.subheader('<Redes Neuronales>')
            arr = algorito_redes_neuronales(uploaded_file.name, Ingrese_x,Ingrese_y)
            st.text('Prediccion con X_test: ')
            st.success(arr[0])
            st.text('Prediccion con Entrada: ')
            st.success(arr[1])
            st.text('Prediccion con Matriz de confucion: ')
            st.success(arr[2])
            st.text('Clasificacion por reporte: ')
            st.success(arr[3])

        elif model =='Arbol':
            st.subheader('<Arbol de decision>')
            algoritmo_tree(uploaded_file.name, Ingrese_x)
            path = './data/' + uploaded_file.name + '.png'
            image = Image.open(path)
            st.image(image,caption='Clasificador de árboles de decisión')

        elif model =='Gauss':
            st.subheader('<Clasificador Gaussiano>')
            predict = algoritmo_Gauss(uploaded_file.name, Ingrese_x, Ingrese_y,Ingrese_z)
            st.text('Prediccion: ')
            st.success(predict)

            path = './data/' + uploaded_file.name + '.png'
            image = Image.open(path)
            st.image(image,caption='Clasificador Gaussiano')

        elif  model == 'Regresion Polinomial':
            st.subheader('<Regresion Polinomial>')
            predict = algorito_polinomial(uploaded_file.name,int(Ingrese_x),int(Ingrese_y),int(Ingrese_z))
            st.text('Prediccion: ')
            st.success(predict)
  
            path = './data/' + uploaded_file.name + '.png'
            image = Image.open(path)
            st.image(image,caption='Regresion Polinomial')

            
if __name__ == '__main__':
    main()