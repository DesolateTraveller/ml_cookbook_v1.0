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
st.set_page_config(page_title="FE CookBook | v0.4",
                   layout="wide", 
                    page_icon="📊",               
                   initial_sidebar_state="auto")
#----------------------------------------
st.markdown(
    """
    <style>
    .title-large {
        text-align: center;
        font-size: 35px;
        font-weight: bold;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-small {
        text-align: center;
        font-size: 20px;
        background: linear-gradient(to left, red, orange, blue, indigo, violet);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    <div class="title-large">Feature Engineering Cookbook</div>
    <div class="title-small">Version : 0.4</div>
    """,
    unsafe_allow_html=True
)

#----------------------------------------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #F0F2F6;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
        z-index: 100;
    }
    .footer p {
        margin: 0;
    }
    .footer .highlight {
        font-weight: bold;
        color: blue;
    }
    </style>

    <div class="footer">
        <p>© 2025 | Created by : <span class="highlight">Avijit Chakraborty</span> | <a href="mailto:avijit.mba18@gmail.com"> 📩 </a></p> <span class="highlight">Thank you for visiting the app | Unauthorized uses or copying is strictly prohibited | For best view of the app, please zoom out the browser to 75%.</span>
    </div>
    """,
    unsafe_allow_html=True)

#---------------------------------------------------------------------------------------------------------------------------------
### Functions & Definitions
#---------------------------------------------------------------------------------------------------------------------------------

@st.cache_data(ttl="2h")
def encode_features(data, encoder):
    if encoder == 'Label Encoder':
        encoder = LabelEncoder()
        encoded_data = data.apply(encoder.fit_transform)
    elif encoder == 'One-Hot Encoder':
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_data = pd.DataFrame(encoder.fit_transform(data), columns=encoder.get_feature_names(data.columns))
    return encoded_data

@st.cache_data(ttl="2h")
def scale_features(data, scaler):
    if scaler == 'Standard Scaler':
        scaler = StandardScaler()
    elif scaler == 'Min-Max Scaler':
        scaler = MinMaxScaler()
    elif scaler == 'Robust Scaler':
        scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)
    return scaled_df

@st.cache_data(ttl="2h")
def random_feature_sampling(df, num_features):
    sampled_features = df.sample(n=num_features, axis=1)
    return sampled_features

@st.cache_data(ttl="2h")        
def plot_feature_correlation(df):
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(25,25))
    ax = sns.heatmap(corr_matrix, annot=corr_matrix.rank(axis="columns"), 
                     cmap="coolwarm", linewidth=.5,fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    st.pyplot(fig)

@st.cache_data(ttl="2h")
def check_missing_values(data):
    missing_values = data.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    return missing_values 

@st.cache_data(ttl="2h")
def check_outliers(data):
                                # Assuming we're checking for outliers in numerical columns
                                numerical_columns = data.select_dtypes(include=[np.number]).columns
                                outliers = pd.DataFrame(columns=['Column', 'Number of Outliers'])

                                for column in numerical_columns:
                                    Q1 = data[column].quantile(0.25)
                                    Q3 = data[column].quantile(0.75)
                                    IQR = Q3 - Q1

                                    # Define a threshold for outliers
                                    threshold = 1.5

                                    # Find indices of outliers
                                    outliers_indices = ((data[column] < Q1 - threshold * IQR) | (data[column] > Q3 + threshold * IQR))

                                    # Count the number of outliers
                                    num_outliers = outliers_indices.sum()
                                    outliers = outliers._append({'Column': column, 'Number of Outliers': num_outliers}, ignore_index=True)

                                return outliers


@st.cache_data(ttl="2h")
def handle_numerical_missing_values(data, numerical_strategy):
    imputer = SimpleImputer(strategy=numerical_strategy)
    numerical_features = data.select_dtypes(include=['number']).columns
    data[numerical_features] = imputer.fit_transform(data[numerical_features])
    return data

@st.cache_data(ttl="2h")
def handle_categorical_missing_values(data, categorical_strategy):
    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
    categorical_features = data.select_dtypes(exclude=['number']).columns
    data[categorical_features] = imputer.fit_transform(data[categorical_features])
    return data                                
#---------------------------------------------------------------------------------------------------------------------------------
### Main App
#---------------------------------------------------------------------------------------------------------------------------------

#stats_expander = st.expander("**Knowledge**", expanded=False)
#with stats_expander: 
with st.popover("**:red[Knowledge]**",disabled=False, use_container_width=True,help="Tune the hyperparameters whenever required"):
        
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
        
        -----------------------------------------------------------------------------------------------------------------------------   
        
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

st.markdown(
        """
        <style>
        .centered-info {display: flex; justify-content: center; align-items: center; 
                        font-weight: bold; font-size: 15px; color: #007BFF; 
                        padding: 5px; background-color: #FFFFFF;  border-radius: 5px; border: 1px solid #007BFF;
                        margin-top: 0px;margin-bottom: 5px;}
        .stMarkdown {margin-top: 0px !important; padding-top: 0px !important;}                       
        </style>
        """,unsafe_allow_html=True,)

#----------------------------------------

col1, col2 = st.columns((0.15,0.85))
with col1:
    with st.container(border=True):

        file = st.file_uploader("**:blue[Choose a file]**",type=["xlsx","csv"],accept_multiple_files=True,key=0)
        if file is not None:
            #st.success("Data loaded successfully!")
            df = pd.DataFrame()
            for file in file:
                df = pd.read_csv(file)

                with col2:
                
                    #---------------------------------------------------------------------------------------------------------------------------------
                    ### Content
                    #---------------------------------------------------------------------------------------------------------------------------------

                    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["**Identification**","**Removal**","**Visualization**","**Correleation**","**Cleaning**","**Encoding**","**Scalling**","**Sampling**","**Selection**"])

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Import & Identification
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab1:
                        with st.container(border=True):
        
                            st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Characteristics</span></div>',unsafe_allow_html=True,)

                            st.table(df.head())
                            st.divider() 
              
                            st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Summary</span></div>',unsafe_allow_html=True,) 
                            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

                            col1.metric('**input values (rows)**', df.shape[0], help='number of rows in the dataframe')
                            col2.metric('**variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
                            col3.metric('**numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
                            col4.metric('**categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')
                            col5.metric('**missing values**', df.isnull().sum().sum(), help='Total missing values in the dataset')
                            col6.metric('**unique categorical values**', sum(df.select_dtypes(include=['object']).nunique()), help='Sum of unique values in categorical variables')

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Removal
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab2:
                        with st.container(border=True):
                            
                            if st.checkbox("**🗑️ Feature Drop**"):
                                feature_to_drop = st.selectbox("**Select Feature to Drop**", df.columns)
                                if feature_to_drop:
                                    if st.button("Apply", key="delete"):
                                        st.session_state.delete_features = True
                                        st.session_state.df = df.drop(feature_to_drop, axis=1)
                                
                                st.divider()
                                st.table(st.session_state.df.head(2))
                                st.download_button("**Download Deleted Data**", st.session_state.df.to_csv(index=False), file_name="deleted_data.csv")

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Visualization
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab3:   
                        with st.container(border=True):   

                            pyg_html = pyg.to_html(df,env='streamlit', return_html=True)
                            components.html(pyg_html, height=1000, scrolling=True)

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Correleation
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab4:      
                        with st.container(border=True): 

                            for feature in df.columns: 
                                if df[feature].dtype == 'object': 
                                    print('\n')
                                    print('feature:',feature)
                                    print(pd.Categorical(df[feature].unique()))
                                    print(pd.Categorical(df[feature].unique()).codes)
                                    df[feature] = pd.Categorical(df[feature]).codes
                            plot_feature_correlation(df)

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Cleaning
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab5:   
                        with st.container(border=True): 

                            st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Missing Values</span></div>',unsafe_allow_html=True,)
                            col1, col2 = st.columns((0.2,0.8))

                            with col1:

                                missing_values = check_missing_values(df)

                                if missing_values.empty:
                                    st.success("**No missing values found!**")
                                else:
                                    st.warning("**Missing values found!**")
                                    st.write("**Number of missing values:**")
                                    st.table(missing_values)

                            with col2:        

                                numerical_strategies = ['mean', 'median', 'most_frequent']
                                categorical_strategies = ['constant','most_frequent']
                                
                                st.write("**Missing Values Treatment:**")
                                selected_numerical_strategy = st.selectbox("**Select a strategy for treatment : Numerical variables**", numerical_strategies)
                                selected_categorical_strategy = st.selectbox("**Select a strategy for treatment : Categorical variables**", categorical_strategies)  
                                
                                cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                #cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                st.table(cleaned_df.head(2))

                                st.download_button("**Download Treated Data**", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                            st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Duplicate Values</span></div>',unsafe_allow_html=True,) 
                            if st.checkbox("Show Duplicate Values"):
                                if missing_values.empty:
                                    st.table(df[df.duplicated()].head(2))
                                else:
                                    st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                            st.markdown('<div class="centered-info"><span style="margin-left: 10px;">Outliers</span></div>',unsafe_allow_html=True,)
                
                            if missing_values.empty:
                                df = df.copy()
                            else:
                                df = cleaned_df.copy()

                            col1, col2 = st.columns((0.2,0.8))

                            with col1:
                                outliers = check_outliers(df)
                                if outliers.empty:
                                    st.success("**No outliers found!**")
                                else:
                                    st.warning("**Outliers found!**")
                                    st.write("**Number of outliers:**")
                                    st.table(outliers)
                    
                            with col2:
                        
                                treatment_option = st.selectbox("**Select a treatment option:**", ["Cap Outliers","Drop Outliers", ])
                                if treatment_option == "Drop Outliers":
                                    df = df[~outliers['Column'].isin(outliers[outliers['Number of Outliers'] > 0]['Column'])]
                                    st.success("Outliers dropped. Preview of the cleaned dataset:")
                                    st.write(df.head())

                                elif treatment_option == "Cap Outliers":
                                    df = df.copy()
                                    for column in outliers['Column'].unique():
                                        Q1 = df[column].quantile(0.25)
                                        Q3 = df[column].quantile(0.75)
                                        IQR = Q3 - Q1
                                        threshold = 1.5

                                        df[column] = np.where(df[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df[column])
                                        df[column] = np.where(df[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df[column])

                                        st.success("Outliers capped. Preview of the capped dataset:")
                                        st.write(df.head())

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Encoding
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab6:   
                        with st.container(border=True): 
                                 
                            encoding_methods = ['Label Encoder', 'One-Hot Encoder']
                            selected_encoder = st.selectbox("**Select a feature encoding method**", encoding_methods)
                    
                            encoded_df = encode_features(df, selected_encoder)
                            st.table(encoded_df.head())  

                            st.divider()
                            st.download_button("**Download Encoded Data**", encoded_df.to_csv(index=False), file_name="encoded_data.csv")

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Scalling
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab7: 
                        with st.container(border=True):  

                            scaling_methods = ['Standard Scaler', 'Min-Max Scaler', 'Robust Scaler']
                            selected_scaler = st.selectbox("**Select a feature scaling method**", scaling_methods)

                            if st.button("**Apply Feature Scalling**", key='f_scl'):

                                st.divider()
                                scaled_df = scale_features(encoded_df, selected_scaler)
                                st.table(scaled_df.head())

                                st.divider()
                                st.download_button("**Download Scaled Data**", scaled_df.to_csv(index=False), file_name="scaled_data.csv")

                            else:
                                df = df.copy()
                                st.table(df.head())

                                st.divider()
                                st.download_button("**Download Original Data**", df.to_csv(index=False), file_name="original_data.csv")

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Sampling
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab8: 
                        with st.container(border=True): 

                            num_features = st.number_input("Number of Features to Sample", min_value=1, step=1, value=1)
                            sampled_features = random_feature_sampling(df, num_features)
                            st.write(sampled_features.head())

                            st.divider()
                            st.download_button("**Download Sampled Data**", df.to_csv(index=False), file_name="sampled_data.csv")

                    #---------------------------------------------------------------------------------------------------------------------------------
                    ## Feature Selection
                    #---------------------------------------------------------------------------------------------------------------------------------

                    with tab9: 
                        with st.container(border=True): 

                            target_variable = st.multiselect("**Target (Dependent) Variable**", df.columns)


        else:
            st.warning("Please upload a excel or csv file.")
