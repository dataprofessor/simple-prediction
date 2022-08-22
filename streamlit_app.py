# Simple Prediction App
# By Chanin Nantasenamat (Data Professor)
# https://youtube.com/dataprofessor

# Importing requisite libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page configuration
st.set_page_config(
     page_title='Simple Prediction App',
     page_icon='🎈',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('🎈 Simple Prediction App')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv')

# Input widgets
st.sidebar.subheader('Input features')
#sepal_length = st.sidebar.slider('Sepal length', df['Sepal.Length'].min(), df['Sepal.Length'].max(), df['Sepal.Length'].median())
#sepal_width = st.sidebar.slider('Sepal width', df['Sepal.Width'].min(), df['Sepal.Width'].max(), df['Sepal.Width'].median())
#petal_length = st.sidebar.slider('Petal length', df['Petal.Length'].min(), df['Petal.Length'].max(), df['Petal.Length'].median())
#petal_width = st.sidebar.slider('Petal width', df['Petal.Width'].min(), df['Petal.Width'].max(), df['Petal.Width'].median())

# Separate to X and y
X = df.drop('Species', axis=1)
y = df.Species

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
rf = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Apply model to make predictions
#y_pred = rf.predict([sepal_length, sepal_width, petal_length, petal_width])

# Print prediction results
#st.write(sepal_length)
#st.write(y_pred)
st.write(df['Sepal.Length'].min())
