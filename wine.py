####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#####
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
####
st.set_page_config(layout = 'wide', initial_sidebar_state = 'expanded')

####
st.sidebar.title('Wine Dataset')
st.sidebar.write('------')
st.sidebar.header('Navigation')
nav = st.sidebar.radio('', ['Introduction', 'Exploratory Data Analysis', 'Graphing and Modelling'])
st.sidebar.write('------')
####

data = load_wine()
X,Y = load_wine(return_X_y = True)
df = pd.DataFrame(X, columns = [data.feature_names])
df['labels'] = Y

feat_cols = df[data.feature_names]
targ_cols = df[['labels']]

cols = ['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids','nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280/od315_of_diluted_wines','proline']
####
X_train, X_test, Y_train, Y_test = train_test_split(feat_cols, targ_cols, random_state = 42, test_size = 0.20)


if nav == 'Introduction':
    st.title('Wine Classification')
    st.write('------')
    st.write(data.DESCR)
    st.write('------')

if nav == 'Exploratory Data Analysis':
    st.title('Wine Classification')
    st.write('------')
    st.sidebar.subheader('Options')
    eda = st.sidebar.radio('', ['Raw Data','Head', 'Tail', 'Statistical Summary', 'Columns and Shape', 'Tools and Packages','Correlation'])
    if eda == 'Raw Data':
        st.subheader('Raw Data')
        st.write(df)
        st.success('Loading success!')
        st.write('------')
    if eda == 'Head':
        st.subheader('Data Head')
        st.write(df.head())
        st.success('Loading success!')
        st.write('------')
    if eda == 'Tail':
        st.subheader('Data Tail')
        st.write(df.tail())
        st.success('Loading success!')
        st.write('------')
    if eda == 'Statistical Summary':
        st.subheader('Statistical Summary')
        st.write(df.describe())
        st.success('Loading success!')
        st.write('------')
    if eda == 'Columns and Shape':
        st.subheader('Columns')
        st.write(data.feature_names)
        st.write('------')
        st.subheader('Shape')
        st.write('The data has: ', df.shape[0], ' Number of Rows.')
        st.write('The data has: ', df.shape[1], ' Number of Columns.')
        st.success('Loading success!')
        st.write('------')
    if eda == 'Correlation':
        st.subheader('Correlation')
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.heatmap(df.corr(), annot=True)
        st.pyplot(fig)
        st.success('Loading success!')
        st.write('------')

    if eda == 'Tools and Packages':
        st.subheader('Tools and Packages')
        st.markdown('''
        * [Pandas](https://pandas.pydata.org/docs/)
        * [Numpy](https://numpy.org/doc/)
        * [Matplotlib](https://matplotlib.org/3.3.3/contents.html)
        * [Seaborn](https://seaborn.pydata.org/)
        * [Sci-Kit Learn](https://scikit-learn.org/stable/)
        * [Streamlit](https://docs.streamlit.io/en/stable/) 
        * [Streamlit Cheat Sheet](https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py)
        ''')
        st.write('------')
####

if nav == 'Graphing and Modelling':
    st.title('Wine Classification')
    st.write('------')
    st.sidebar.subheader('Choose Model Algorithm')
    opt = st.sidebar.selectbox('', ['Default','K-Neighbors Classifier', 'Support Vector Classifier', 'Random Forest Classifier'])
    if opt == 'K-Neighbors Classifier':
        st.sidebar.subheader('Number of Neighbors')
        ax = st.sidebar.slider('', 2, 100)
        st.sidebar.subheader('Weights')
        bx = st.sidebar.selectbox('', ['distance', 'uniform'])
        st.sidebar.subheader('Algorithm')
        cx = st.sidebar.selectbox('', ['auto','kd_tree', 'ball_tree', 'brute'])
        knn_model = KNeighborsClassifier(n_neighbors = ax, weights = bx, algorithm = cx).fit(X_train, Y_train)
        y_predict = knn_model.predict_proba(X_test)
        st.header('K Nearest Neighbor Algorithm')
        st.write('**Results**')
        st.write('**Score:** ', knn_model.score(X_test, Y_test))
        st.write('**Classification Report**')
        st.write(classification_report(Y_test, knn_model.predict(X_test)))

        st.write('**Predicted Labels**')
        if st.checkbox('Prediction'):
            st.write('*First 10 items predicted*')
            a_score = []
            for i in range(19):
                a_score.append(Y_test.values[i])
            df_proba = pd.DataFrame(y_predict[:19], columns=['0', '1', '2'])
            df_proba['Actual'] = a_score
            st.write(df_proba)

        st.write('**Optimization**')
        if st.checkbox('Optimization'):
            cv_score = []
            n_neighbors = np.arange(2, 100)

            for i in n_neighbors:
                knn_model = KNeighborsClassifier(n_neighbors=i)
                scores = cross_val_score(knn_model, X_train, Y_train, cv=5, scoring='accuracy')
                cv_score.append(np.mean(scores))

            df_score = pd.DataFrame(n_neighbors, columns=['neighbors']).set_index('neighbors')
            df_score['cv_score'] = cv_score

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.lineplot(x=df_score.index, y=df_score['cv_score'])

            st.pyplot(fig)
            st.success('Loading success!')

####
    if opt == 'Random Forest Classifier':
        st.sidebar.write('**Additional Option**')
        ax = st.sidebar.selectbox('', ['Vanilla', 'Recursive Feature Elimination'])
        if ax == 'Vanilla':
            feat_cols = df[data.feature_names]
            targ_cols = df[['labels']]
            st.sidebar.write('**Criterion**')
            bx = st.sidebar.selectbox('',['gini','entropy'])
            st.sidebar.write('**Max Depth**')
            cx = st.sidebar.slider('', 2,20)
            model_rf = RandomForestClassifier(criterion = 'gini', max_depth = 2).fit(X_train, Y_train)
            y_pred = model_rf.predict(X_test)
            y_pred_proba = model_rf.predict_proba(X_test)

            st.header('Random Forest Classifier Algorithm')
            st.write('**Results**')
            st.write('**Score:** ', model_rf.score(X_test, Y_test))
            st.write('**Classification Report**')
            st.write(classification_report(Y_test, model_rf.predict(X_test)))

            st.write('**Predicted Labels**')
            if st.checkbox('Prediction'):
                st.write('*First 10 items predicted*')
                a_score = []
                for i in range(19):
                    a_score.append(Y_test.values[i])
                df_proba = pd.DataFrame(y_pred_proba[:19], columns=['0', '1', '2'])
                df_proba['Actual'] = a_score
                st.write(df_proba[:10])
                st.success('Loading success!')

            st.write('**Feature Importance**')
            if st.checkbox('Feature Importance'):
                importance = model_rf.feature_importances_
                plt.figure(figsize=(16, 10))
                fig, ax = plt.subplots()
                ax.barh(data.feature_names, importance, color='red')
                plt.yticks(data.feature_names, fontsize=16)
                ax.set_yticklabels(data.feature_names)
                st.pyplot(fig)
                st.success('Loading success!')

        if ax == 'Recursive Feature Elimination':
            feat_cols = df[data.feature_names]
            targ_cols = df[['labels']]
            st.sidebar.write('**Criterion**')
            bx = st.sidebar.selectbox('', ['gini', 'entropy'])
            st.sidebar.write('**Max Depth**')
            cx = st.sidebar.slider('', 2, 20)
            model_rfe = RFE(RandomForestClassifier(criterion=bx, max_depth=cx), n_features_to_select = 13, step = 1).fit(X_train, Y_train)
            y_pred = model_rfe.predict(X_test)
            y_pred_proba = model_rfe.predict_proba(X_test)

            st.header('Random Forest Classifier Algorithm with Recursive Feature Elimination')
            st.write('**Results**')
            st.write('**Score:** ', model_rfe.score(X_test, Y_test))
            st.write('**Classification Report**')
            st.write(classification_report(Y_test, model_rfe.predict(X_test)))

            st.write('**Predicted Labels**')
            if st.checkbox('Prediction'):
                st.write('*First 10 items predicted*')
                a_score = []
                for i in range(19):
                    a_score.append(Y_test.values[i])
                df_proba = pd.DataFrame(y_pred_proba[:19], columns=['0', '1', '2'])
                df_proba['Actual'] = a_score
                st.write(df_proba[:10])
                st.success('Loading success!')

            st.write('**Feature Importance**')
            if st.checkbox('Feature Importance'):
                importance = model_rfe.estimator_.feature_importances_
                plt.figure(figsize=(16, 10))
                fig, ax = plt.subplots()
                ax.barh(data.feature_names, importance, color='red')
                plt.yticks(data.feature_names, fontsize=16)
                ax.set_yticklabels(data.feature_names)
                st.pyplot(fig)
                st.success('Loading success!')

####
    if opt == 'Support Vector Classifier':
        st.sidebar.write('**Kernel**')
        ax = st.sidebar.selectbox('',['rbf','linear','poly'])
        st.sidebar.write('**gamma**')
        bx = st.sidebar.selectbox('', ['auto', 'scale'])
        st.sidebar.write('**C - Value**')
        cx = st.sidebar.slider('c-value',1, 10)

        if ax == 'linear':
            model_svm = SVC(kernel='linear', random_state=42, gamma=bx, C = cx).fit(X_train, Y_train)
            y_predict = model_svm.predict(X_test)
            st.header('Support Vector Classifier with Linear Kernel')
            st.write('**Results**')
            st.write('**Score:** ', model_svm.score(X_test, Y_test))
            st.write('**Classification Report**')
            st.write(classification_report(Y_test, model_svm.predict(X_test)))
            if st.checkbox('Predicted Values'):
                df1 = pd.DataFrame(Y_test.values, columns = ['Actual'])
                df1['Predicted'] = y_predict
                st.write(df1)

        if ax == 'poly':
            st.sidebar.write('**Degree**')
            dx = st.sidebar.slider('degree',1,10)
            model_svm = SVC(kernel='poly', random_state=42, gamma=bx, degree = dx, C = cx).fit(X_train, Y_train)
            y_predict = model_svm.predict(X_test)
            st.header('Support Vector Classifier with Polynomial Kernel')
            st.write('**Results**')
            st.write('**Score:** ', model_svm.score(X_test, Y_test))
            st.write('**Classification Report**')
            st.write(classification_report(Y_test, model_svm.predict(X_test)))
            if st.checkbox('Predicted Values'):
                df1 = pd.DataFrame(Y_test.values, columns = ['Actual'])
                df1['Predicted'] = y_predict
                st.write(df1)

        if ax == 'rbf':
            model_svm = SVC(kernel='rbf', random_state=42, gamma=bx, C = cx).fit(X_train, Y_train)
            y_predict = model_svm.predict(X_test)
            st.header('Support Vector Classifier with Polynomial Kernel')
            st.write('**Results**')
            st.write('**Score:** ', model_svm.score(X_test, Y_test))
            st.write('**Classification Report**')
            st.write(classification_report(Y_test, model_svm.predict(X_test)))
            if st.checkbox('Predicted Values'):
                df1 = pd.DataFrame(Y_test.values, columns = ['Actual'])
                df1['Predicted'] = y_predict
                st.write(df1)

