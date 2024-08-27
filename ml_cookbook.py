
#---------------------------------------------------------------------------------------------------------------------------------
### Authenticator
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False)
#---------------------------------------------------------------------------------------------------------------------------------
### Template Graphics
#---------------------------------------------------------------------------------------------------------------------------------
import streamlit.components.v1 as components
#---------------------------------------------------------------------------------------------------------------------------------
### Import Libraries
#---------------------------------------------------------------------------------------------------------------------------------
from streamlit_extras.stoggle import stoggle
#from ydata_profiling import ProfileReport
#from streamlit_pandas_profiling import st_profile_report
#----------------------------------------
import os
import time
import warnings
from random import randint
warnings.filterwarnings("ignore")
#----------------------------------------
import json
import base64
import codecs
import holidays
import itertools
from datetime import datetime, timedelta, date
#----------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#----------------------------------------
import altair as alt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
#----------------------------------------
import pygwalker as pyg
#----------------------------------------
from sklearn.impute import SimpleImputer, KNNImputer
#----------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#----------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, f_regression, chi2, VarianceThreshold
#----------------------------------------
from sklearn.metrics import accuracy_score
#---------------------------------------------------------------------------------------------------------------------------------
### Title and description for your Streamlit app
#---------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="FE CookBook | v0.3",
                   layout="wide", 
                    page_icon="📊",               
                   initial_sidebar_state="auto")
#----------------------------------------
st.title(f""":rainbow[Feature Engineering CookBook]""")
st.markdown(
    '''
    Created by | <a href="mailto:avijit.mba18@gmail.com">Avijit Chakraborty</a> ( 📑 [Resume](https://resume-avijitc.streamlit.app/) | :bust_in_silhouette: [LinkedIn](https://www.linkedin.com/in/avijit2403/) | :computer: [GitHub](https://github.com/DesolateTraveller) ) |
    for best view of the app, please **zoom-out** the browser to **75%**.
    ''',
    unsafe_allow_html=True)
st.info('**Disclaimer : :blue[Thank you for visiting the app] | Unauthorized uses or copying of the app is strictly prohibited | Please expand the below :blue[Knowledge] tab to know more and click the :blue[sidebar] to follow the instructions to start the applications.**', icon="ℹ️")
#----------------------------------------
# Set the background image
#st.divider()

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

stats_expander = st.expander("**Knowledge**", expanded=False)
with stats_expander: 
        
        st.info('''
        **Machine Learning**

        Machine learning is a field of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. Here are some key points about machine learning:

        - It involves developing algorithms that can analyze data and make predictions or decisions without being specifically told what to do. The algorithms "learn" by detecting patterns in data.
        - Machine learning algorithms build a mathematical model from sample data, known as "training data", to make predictions or decisions without being explicitly programmed to perform the task.

        In summary, machine learning allows computers to learn patterns from data to make decisions and predictions, rather than requiring explicit programming for every task. It allows for systems and services that adapt and evolve over time.
            
        Further, it can be subdivided into the following :
        - 1 | **Supervised Learning** 
        - 2 | **Un supervised Learning**
        - 3 | **Semi supervised Learning** 
        - 4 | **Reinforcement Learning**             
            
        **Feature engineering** 
                    
          It is a very important part of dataset preparation. 
          During the process, It create a set of attributes (input features) that represent various behavior patterns towards the target features. 
          In a broad sense, features are the measurable characteristics of observations that a ML model takes into account to predict outcomes.
    
        A typical **Exploratory Data Analysis (EDA)** consists of the following steps:
    
        - 1 | **Feature Import** :               Import and read the collected dataset.
        - 2 | **Feature Removal** :              Remove unwanted features.
        - 3 | **Feature Identification** :       Variable type identifications.
        - 4 | **Feature Visualization** :        Graphical representation of univariate and bi variate analysis.
        - 5 | **Feature Scalling** :             Rescalling the features.  
        - 6 | **Feature Cleaning** :             Checking missing values, duplicate values and outlier and do the required treatment. 
        - 7 | **Feature Encoding** :             Conversion of the categorical variable into continuous variable.
        - 8 | **Feature Selection** :            Checking of multi collinearity, p values, recursive feature elimination (via chi-square method).
        - 9 | **Feature Sampling** :             Redefine the datasets to increase the performance.
        ''')

#----------------------------------------
#   
file = st.file_uploader("**:blue[Choose a file]**",type=["xlsx","csv"],accept_multiple_files=True,key=0)
if file is not None:


#---------------------------------------------------------------------------------------------------------------------------------
### Content
#---------------------------------------------------------------------------------------------------------------------------------

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["**Identification**","**Removal**","**Visualization**","**Correleation**","**Cleaning**","**Encoding**","**Scalling**","**Sampling**","**Selection**"])


#---------------------------------------------------------------------------------------------------------------------------------
## Feature Import & Identification
#---------------------------------------------------------------------------------------------------------------------------------

    with tab1:
        
        df = pd.DataFrame()
        for file in file:
            df = pd.read_csv(file)

            st.subheader("Feature Information", divider='blue') 

            st.table(df.head(3))
            st.divider() 
              
            st.subheader("Characteristics", divider='blue') 
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            col1.metric('**Number of input values (rows)**', df.shape[0], help='number of rows in the dataframe')
            col2.metric('**Number of variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
            col3.metric('**Number of numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
            col4.metric('**Number of categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
