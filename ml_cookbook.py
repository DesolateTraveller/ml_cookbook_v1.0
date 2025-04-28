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
                    ax = sns.heatmap(corr_matrix, 
                                     annot=corr_matrix.rank(axis="columns"), 
                                     cmap="coolwarm", 
                                     linewidth=.5,
                                     fmt=".2f")
                    plt.title("Feature Correlation Heatmap")
                    plt.xticks(rotation=45)
                    plt.yticks(rotation=45)
                    st.pyplot(fig)

                plot_feature_correlation(df)

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Cleaning
#---------------------------------------------------------------------------------------------------------------------------------

        with tab5:   

                
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

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Encoding
#---------------------------------------------------------------------------------------------------------------------------------

        with tab6:   
     
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

            num_features = st.number_input("Number of Features to Sample", min_value=1, step=1, value=1)
            sampled_features = random_feature_sampling(df, num_features)
            st.write(sampled_features.head())

            st.divider()
            st.download_button("**Download Sampled Data**", df.to_csv(index=False), file_name="sampled_data.csv")

#---------------------------------------------------------------------------------------------------------------------------------
## Feature Selection
#---------------------------------------------------------------------------------------------------------------------------------

        with tab9:  

            target_variable = st.multiselect("**Target (Dependent) Variable**", df.columns)

            with col2:

                st.subheader("Feature Selection",divider='blue') 
                #----------------------------------------
                for feature in df.columns: 
                    if df[feature].dtype == 'object': 
                        print('\n')
                        print('feature:',feature)
                        print(pd.Categorical(df[feature].unique()))
                        print(pd.Categorical(df[feature].unique()).codes)
                        df[feature] = pd.Categorical(df[feature]).codes
                #----------------------------------------

                fsel_method = st.selectbox("**Select Feature Selection Method**", ["VIF", "SelectKBest", "VarianceThreshold" ])
                st.divider()
                if fsel_method == "VIF":   
                    
                    st.markdown("**Method 1 : Checking VIF Values**")
                    vif_threshold = st.number_input("**VIF Threshold**", 1.5, 10.0, 5.0)
                    @st.cache_data(ttl="2h")
                    def calculate_vif(data):
                        X = data.values
                        vif_data = pd.DataFrame()
                        vif_data["Variable"] = data.columns
                        vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
                        vif_data = vif_data.sort_values(by="VIF", ascending=False)
                        return vif_data

                    # Function to drop variables with VIF exceeding the threshold
                    @st.cache_data(ttl="2h")
                    def drop_high_vif_variables(data, threshold):
                        vif_data = calculate_vif(data)
                        high_vif_variables = vif_data[vif_data["VIF"] > threshold]["Variable"].tolist()
                        data = data.drop(columns=high_vif_variables)
                        return data
                                       
                    st.markdown(f"Iterative VIF Thresholding (Threshold: {vif_threshold})")
                    #X = df.drop(columns = target_variable)
                    vif_data = drop_high_vif_variables(df, vif_threshold)
                    vif_data = vif_data.drop(columns = target_variable)
                    selected_features = vif_data.columns
                    st.markdown("**Selected Features (considering VIF values in ascending orders)**")
                    st.table(selected_features)

                if fsel_method == "SelectKBest":  

                    st.markdown("**Method 2 : Checking Selectkbest Method**")          
                    method = st.selectbox("**Select Feature Selection Method**", ["f_classif", "f_regression", "chi2", "mutual_info_classif"])
                    num_features_to_select = st.slider("**Select Number of Independent Features**", min_value=1, max_value=len(df.columns), value=5)

                    # Perform feature selection
                    if "f_classif" in method:
                            feature_selector = SelectKBest(score_func=f_classif, k=num_features_to_select)

                    elif "f_regression" in method:
                            feature_selector = SelectKBest(score_func=f_regression, k=num_features_to_select)

                    elif "chi2" in method:
                            # Make sure the data is non-negative for chi2
                            df[df < 0] = 0
                            feature_selector = SelectKBest(score_func=chi2, k=num_features_to_select)

                    elif "mutual_info_classif" in method:
                            # Make sure the data is non-negative for chi2
                            df[df < 0] = 0
                            feature_selector = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)

                    X = df.drop(columns = target_variable)  # Adjust 'Target' to your dependent variable
                    y = df[target_variable]  # Adjust 'Target' to your dependent variable
                    X_selected = feature_selector.fit_transform(X, y)

                    # Display selected features
                    selected_feature_indices = feature_selector.get_support(indices=True)
                    selected_features_kbest = X.columns[selected_feature_indices]
                    st.markdown("**Selected Features (considering values in 'recursive feature elimination' method)**")
                    st.table(selected_features_kbest)

                if fsel_method == "VarianceThreshold":  

                    st.markdown("**Method 3 : Checking VarianceThreshold Method**")    
                    def variance_threshold_feature_selection(df, threshold):

                        X = df.drop(columns=df[target_variable])  
                        selector = VarianceThreshold(threshold=threshold)
                        X_selected = selector.fit_transform(X)

                        selected_feature_indices = selector.get_support(indices=True)

                        selected_feature_names = X.columns[selected_feature_indices]
                        selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

                        return selected_df
                    
                    threshold = st.number_input("Variance Threshold", min_value=0.0, step=0.01, value=0.0)
                    selected_df = variance_threshold_feature_selection(df, threshold)
                    st.write(selected_df.head())


else:
    st.warning("Please upload a excel or csv file.")
