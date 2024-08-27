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
    df = pd.DataFrame()
    for file in file:
        df = pd.read_csv(file)

#---------------------------------------------------------------------------------------------------------------------------------
### Content
#---------------------------------------------------------------------------------------------------------------------------------

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["**Identification**","**Removal**","**Visualization**","**Correleation**","**Cleaning**","**Encoding**","**Scalling**","**Sampling**","**Selection**"])


#---------------------------------------------------------------------------------------------------------------------------------
## Feature Import & Identification
#---------------------------------------------------------------------------------------------------------------------------------

        with tab1:
        
            st.subheader("Feature Information", divider='blue') 

            st.table(df.head())
            st.divider() 
              
            st.subheader("Characteristics", divider='blue') 
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            col1.metric('**Number of input values (rows)**', df.shape[0], help='number of rows in the dataframe')
            col2.metric('**Number of variables (columns)**', df.shape[1], help='number of columns in the dataframe')     
            col3.metric('**Number of numerical variables**', len(df.select_dtypes(include=['float64', 'int64']).columns), help='number of numerical variables')
            col4.metric('**Number of categorical variables**', len(df.select_dtypes(include=['object']).columns), help='number of categorical variables')

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Removal
#---------------------------------------------------------------------------------------------------------------------------------

        with tab2:

            if st.checkbox("**🗑️ Feature Drop**"):
                feature_to_drop = st.selectbox("**Select Feature to Drop**", df.columns)
                #df_dropped = df.drop(columns=[feature_to_drop])
                if feature_to_drop:
                    #col1, col2, col3 = st.columns([1, 0.5, 1])
                        if st.button("Apply", key="delete"):
                            st.session_state.delete_features = True
                            st.session_state.df = df.drop(feature_to_drop, axis=1)
                                
                            st.divider()
                            st.table(st.session_state.df.head(2))
                            # Download link for treated data
                            st.download_button("**Download Deleted Data**", st.session_state.df.to_csv(index=False), file_name="deleted_data.csv")

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Visualization
#---------------------------------------------------------------------------------------------------------------------------------

        with tab3:      

            pyg_html = pyg.to_html(df,env='streamlit', return_html=True)
            components.html(pyg_html, height=1000, scrolling=True)

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Correleation
#---------------------------------------------------------------------------------------------------------------------------------

        with tab4:      

                #----------------------------------------
                for feature in df.columns: 
                    if df[feature].dtype == 'object': 
                        print('\n')
                        print('feature:',feature)
                        print(pd.Categorical(df[feature].unique()))
                        print(pd.Categorical(df[feature].unique()).codes)
                        df[feature] = pd.Categorical(df[feature]).codes
                #----------------------------------------
                @st.cache_data(ttl="2h")        
                def plot_feature_correlation(df):
                    # Calculate correlation matrix
                    corr_matrix = df.corr()

                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(10,10))
                    ax = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
                    plt.title("Feature Correlation Heatmap")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    st.pyplot(fig)

                plot_feature_correlation(df)

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Cleaning
#---------------------------------------------------------------------------------------------------------------------------------

        with tab4:   

                
                st.subheader("Missing Values Check & Treatment",divider='blue')
                col1, col2 = st.columns((0.2,0.8))

                with col1:
                    @st.cache_data(ttl="2h")
                    def check_missing_values(data):
                            missing_values = data.isnull().sum()
                            missing_values = missing_values[missing_values > 0]
                            return missing_values 
                    missing_values = check_missing_values(df)

                    if missing_values.empty:
                            st.success("**No missing values found!**")
                    else:
                            st.warning("**Missing values found!**")
                            st.write("**Number of missing values:**")
                            st.table(missing_values)

                            with col2:        
                                #treatment_option = st.selectbox("**Select a treatment option**:", ["Impute with Mean","Drop Missing Values", ])
        
                                # Perform treatment based on user selection
                                #if treatment_option == "Drop Missing Values":
                                    #df = df.dropna()
                                    #st.success("Missing values dropped. Preview of the cleaned dataset:")
                                    #st.table(df.head())
            
                                #elif treatment_option == "Impute with Mean":
                                    #df = df.fillna(df.mean())
                                    #st.success("Missing values imputed with mean. Preview of the imputed dataset:")
                                    #st.table(df.head())
                                 
                                # Function to handle missing values for numerical variables
                                @st.cache_data(ttl="2h")
                                def handle_numerical_missing_values(data, numerical_strategy):
                                    imputer = SimpleImputer(strategy=numerical_strategy)
                                    numerical_features = data.select_dtypes(include=['number']).columns
                                    data[numerical_features] = imputer.fit_transform(data[numerical_features])
                                    return data

                                # Function to handle missing values for categorical variables
                                @st.cache_data(ttl="2h")
                                def handle_categorical_missing_values(data, categorical_strategy):
                                    imputer = SimpleImputer(strategy=categorical_strategy, fill_value='no_info')
                                    categorical_features = data.select_dtypes(exclude=['number']).columns
                                    data[categorical_features] = imputer.fit_transform(data[categorical_features])
                                    return data            

                                numerical_strategies = ['mean', 'median', 'most_frequent']
                                categorical_strategies = ['constant','most_frequent']
                                st.write("**Missing Values Treatment:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    selected_numerical_strategy = st.selectbox("**Select a strategy for treatment : Numerical variables**", numerical_strategies)
                                with col2:
                                    selected_categorical_strategy = st.selectbox("**Select a strategy for treatment : Categorical variables**", categorical_strategies)  
                                
                                #if st.button("**Apply Missing Values Treatment**"):
                                cleaned_df = handle_numerical_missing_values(df, selected_numerical_strategy)
                                cleaned_df = handle_categorical_missing_values(cleaned_df, selected_categorical_strategy)   
                                st.table(cleaned_df.head(2))

                                # Download link for treated data
                                st.download_button("**Download Treated Data**", cleaned_df.to_csv(index=False), file_name="treated_data.csv")

                #with col2:

                st.subheader("Duplicate Values Check",divider='blue') 
                if st.checkbox("Show Duplicate Values"):
                    if missing_values.empty:
                        st.table(df[df.duplicated()].head(2))
                    else:
                        st.table(cleaned_df[cleaned_df.duplicated()].head(2))

                #with col4:

                    #x_column = st.selectbox("Select x-axis column:", options = df.columns.tolist()[0:], index = 0)
                    #y_column = st.selectbox("Select y-axis column:", options = df.columns.tolist()[0:], index = 1)
                    #chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(x=x_column,y=y_column)
                    #st.altair_chart(chart, theme=None, use_container_width=True)  

                st.subheader("Outliers Check & Treatment",divider='blue')
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
                
                if missing_values.empty:
                    df = df.copy()
                else:
                    df = cleaned_df.copy()

                col1, col2 = st.columns((0.2,0.8))

                with col1:
                        # Check for outliers
                        outliers = check_outliers(df)

                        # Display results
                        if outliers.empty:
                            st.success("**No outliers found!**")
                        else:
                            st.warning("**Outliers found!**")
                            st.write("**Number of outliers:**")
                            st.table(outliers)
                    
                with col2:
                        # Treatment options
                        treatment_option = st.selectbox("**Select a treatment option:**", ["Cap Outliers","Drop Outliers", ])

                            # Perform treatment based on user selection
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

                                    # Cap outliers
                                    df[column] = np.where(df[column] < Q1 - threshold * IQR, Q1 - threshold * IQR, df[column])
                                    df[column] = np.where(df[column] > Q3 + threshold * IQR, Q3 + threshold * IQR, df[column])

                                    st.success("Outliers capped. Preview of the capped dataset:")
                                    st.write(df.head())
