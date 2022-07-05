import streamlit as st
import pickle
import pandas as pd

#extraer los archivos pickle
with open('lin_reg.pkl','rb') as li:
    lin_reg = pickle.load(li)


with open('log_reg.pkl','rb') as lo:
    log_reg = pickle.load(lo)

with open('svc_model.pkl','rb') as sv:
    svc_model = pickle.load(sv)




def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'VersionColor'
    else:
        return 'Virginica'


def main():
    #titulo
    st.title('Modelamiento de Iris')
    #titulo de side bar
    st.sidebar.header('User input parameters')
    #funcion para poner los parametros en el side bar
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal length',4.3,7.9,5.4)
        sepal_width = st.sidebar.slider('Sepal width',2.0,4.4,3.4)
        pepal_length = st.sidebar.slider('Petal length',1.0,6.9,1.3)
        pepal_width = st.sidebar.slider('Petal length',0.1,2.5,0.2)
        data = {
            'Sepal length' : sepal_length,
            'Sepal width': sepal_width,
            'Petal length': pepal_length,
            'Petal width' : pepal_width,
        }
        features = pd.DataFrame(data,index=[0])
        return features

    df = user_input_parameters()
    #escojer el modelo
    option = ('Linear Regression' , 'Logistic Regression','svm')
    model = st.sidebar.selectbox('which model you like to use', option)

    st.subheader('User input parameters')
    st.subheader(model)
    st.write(df)

#boton
    if st.button('Run'):
        if model == 'Linear Regression':
            st.success(classify(lin_reg.predict(df)))
        elif model == 'Logistic Regression':
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_model.predict(df)))




if __name__ == '__main__':
    main()