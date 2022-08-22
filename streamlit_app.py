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
st_sepal_length = st.sidebar.slider('Sepal length', df['Sepal.Length'].min(), df['Sepal.Length'].max(), df['Sepal.Length'].mean())

# Separate to X and y
X = df.drop('Species', axis=1)
y = df.Species

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st_test_size, random_state=42)

# Model building
rf = RandomForestClassifier(max_depth=2, max_features=st_max_features, n_estimators=st_n_estimators, random_state=42)
rf.fit(X_train, y_train)

# Apply model to make predictions
y_pred = rf.predict(X_train)

# Print prediction results
st.write(st_sepal_length)
st.write(y_pred)
