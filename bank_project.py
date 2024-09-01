import numpy as np
import pandas as pd
import streamlit as st 
from streamlit_option_menu import option_menu 
import seaborn as sns 
import matplotlib.pyplot as plt
from joblib import load

from wordcloud import STOPWORDS, WordCloud

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

import re
import warnings as wr
wr.filterwarnings('ignore')

#from pandas_profiling import ProfileReport
#from ydata_profiling import ProfileReport
import sweetviz as sv
import codecs

from sklearn.preprocessing import StandardScaler
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

from PIL import Image
import io

st.set_option('deprecation.showPyplotGlobalUse', False)
# ***** STREAMLIT PAGE ICON ***** 

icon = Image.open("C:/Users/mahes/Downloads/imageml.png")
# SETTING PAGE CONFIGURATION...........
st.set_page_config(page_title='BANKING PROJECT',page_icon=icon,layout="wide")

html_temp = """
        <div style="background-color:#08457e;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">COMPREHENSIVE BANKING ANALYSIS AND PREDICTION</h1>
        </div>"""

# components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
components.html(html_temp)
style = "<style>h2 {text-align: center;color: #00ced1}</style>"
style1 = "<style>h3 {text-align: left;}</style>"


selected = option_menu(None,
                       options = ["Home","Data View and EDA","ML Prediction"],
                       icons = ["house-door-fill","bar-chart-line-fill","bi-binoculars-fill"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"container": {"width": "100%"},
                               "icon": {"color": "white", "font-size": "24px"},
                               "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                               "nav-link-selected": {"background-color": "#00008b"}})


def home_menu():
        col1,col2 = st.columns(2)
        with col1:
                st.image(Image.open("C:/Users/mahes/OneDrive/Desktop/bank project/Banking.png"),width=650)
                st.video("C:/Users/mahes/OneDrive/Desktop/bank project/main.mp4")
                st.markdown("## Done by : UMAMAHESWARI S")
                st.markdown(style,unsafe_allow_html=True)
                st.markdown("[Githublink](https://github.com/mahes101)")
                
                
        with col2:
                st.title(':blue[COMPREHENSIVE BANKING ANALYSIS]')  
                st.header('[BANKING]')  
                st.markdown(style, unsafe_allow_html=True)    
                st.write("All financial institutions will need to examine each of their businesses to assess where their competitive advantages lie across and within the three core banking activities of balance sheet, transactions, and distribution. And they will need to do so in a world in which technology and AI will play a more prominent role, and against the backdrop of a shifting macroeconomic environment and heightened geopolitical risks.")
                st.markdown(style1, unsafe_allow_html=True)
                st.header('[SKILLS OR TECHNOLOGIES]')
                st.markdown(style, unsafe_allow_html=True)
                st.write("Python scripting, Data Preprocessing, Visualization(Matplolib,Seaborn), EDA, Streamlit, Data Analysis(Pandas),Auto EDA(SWEETVIZ), Kmode Clustering,Classification,Regression(MLAlgorithms(Scikit-learn))")
                st.markdown(style1, unsafe_allow_html=True)
                st.header('[DOMAIN]')
                st.markdown(style, unsafe_allow_html=True)
                st.write("Banking and Financial Industry")
                st.markdown(style1, unsafe_allow_html=True)
               
@ st.cache_data                
def load_data():
        df_air=pd.read_csv("C:/Users/mahes/OneDrive/Desktop/BANK PROJECT/bank.csv")
        df_air=pd.DataFrame(df_air)  
        return df_air
df = load_data()   
df_new = df.copy()
def univariate_analysis(df):
        opt = ["Numerical columns","Categorical columns"]
        option = st.sidebar.selectbox("Select an option",opt)
        
        # Identify numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        if option == "Numerical columns":
                # Perform univariate analysis on numerical columns
                for column in numerical_columns:
                # For continuous variables
                        if len(df[column].unique()) > 10:  # Assuming if unique values > 10, consider it continuous
                                plt.figure(figsize=(8, 6))
                                sns.histplot(df[column], kde=True)
                                plt.title(f'Histogram of {column}')
                                plt.xlabel(column)
                                plt.ylabel('Frequency')
                                plt.show()
                                st.pyplot()
                        else:  # For discrete or ordinal variables
                                plt.figure(figsize=(8, 6))
                                ax = sns.countplot(x=column, data=df)
                                plt.title(f'Count of {column}')
                                plt.xlabel(column)
                                plt.ylabel('Count')
                                
                                # Annotate each bar with its count
                                for p in ax.patches:
                                        ax.annotate(format(p.get_height(), '.0f'), 
                                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                                ha = 'center', va = 'center', 
                                                xytext = (0, 5), 
                                                textcoords = 'offset points')
                                #plt.show()
                                st.pyplot()

        elif option == "Categorical columns":        
                #univariate Analysis for categorical column
                fig, axes = plt.subplots(3, 2, figsize = (18, 18))
                fig.suptitle('Bar plot for all categorical variables in the dataset')
                sns.countplot(ax = axes[0, 0], x = 'Occupation', data = df, color = 'blue', 
                        order = df['Occupation'].value_counts().index)
                sns.countplot(ax = axes[0, 1], x = 'Type_of_Loan', data = df, color = 'blue', 
                        order = df['Type_of_Loan'].value_counts().index)
                sns.countplot(ax = axes[1, 0], x = 'Credit_Mix', data = df, color = 'blue', 
                        order = df['Credit_Mix'].value_counts().index)
                sns.countplot(ax = axes[1, 1], x = 'Payment_of_Min_Amount', data = df, color = 'blue', 
                        order = df['Payment_of_Min_Amount'].value_counts().index)
                sns.countplot(ax = axes[2, 0], x = 'Payment_Behaviour', data = df, color = 'blue', 
                        order = df['Payment_Behaviour'].head(20).value_counts().index)
                sns.countplot(ax = axes[2, 1], x = 'Credit_Score', data = df, color = 'blue', 
                        order = df['Credit_Score'].head(20).value_counts().index)

                axes[1][1].tick_params(labelrotation=45)
                axes[2][0].tick_params(labelrotation=90)
                axes[2][1].tick_params(labelrotation=90);  
                st.pyplot(fig)      
                        
def bivariate_analysis():
        opt1 = ["Numerical columns","Categorical columns"]
        option1 = st.sidebar.selectbox("Select an option",opt1)
        if option1 == "Categorical columns":
                st.title("")
                plt.figure(figsize=(16, 8))
                sns.violinplot(data=df, y="Month", x="Num_of_Loan")
                plt.title('MONTH VS NUM_OF_LOAN')
                st.pyplot()
                
                plt.figure(figsize=(13,17))
                sns.pairplot(data=df.drop(['Month','Credit_Score'],axis=1))
                st.pyplot()
        
                plt.figure(figsize=(16, 8))
                sns.lineplot(data=df, y="Month", x="Credit_Score")
                plt.title('MONTH vs CREDIT_SCORE')
                st.pyplot()
                
                plt.figure(figsize=(16, 8))
                sns.violinplot(data=df, x="Month", y="Total_EMI_per_month")
                plt.title('Month vs Total_EMI_per_month')
                st.pyplot()
                
                plt.figure(figsize=(16, 8))
                sns.barplot(data=df, x="Month", y="Amount_invested_monthly")
                plt.title('Month vs Amount_invested_monthly')
                st.pyplot()
        if option1 == "Numerical columns": 
                #Plotting a pair plot for bivariate analysis
                g = sns.PairGrid(df,vars=['Interest_Rate','Annual_Income','Num_of_Loan','Total_EMI_per_month','Monthly_Balance'])
                #setting color
                g.map_upper(sns.scatterplot, color='crimson')
                g.map_lower(sns.scatterplot, color='limegreen')
                g.map_diag(plt.hist, color='orange')
                #show figure
                st.pyplot()

        
def correlation_plot():
        #setting the figure size and fontsize
        numeric_df = df.select_dtypes(include=['number'])
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar_kws={"shrink": .5})
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.title('Correlation Matrix of Numeric Variables')
        plt.tight_layout()
        plt.show()
        #plt.show()   
        st.pyplot(fig)
def wordcloud_fun(df):
        house_mask = np.array(Image.open('C:/Users/mahes/Downloads/house1.jpg'))
        text = df['Type_of_Loan'].dropna()
        text = text.apply(lambda x: x.lower())
        new_words = ['Good','Standard','bad']
        text_1 = ' '.join(i for i in text)
        stopwords = list(STOPWORDS) + new_words
        wordcloud = WordCloud(stopwords=stopwords, background_color='#1C1C1C', mask=house_mask, colormap = 'twilight_shifted_r').generate(text_1)
        g1 = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(g1)        
        
def data_view_func(df):
    with st.expander("DATA VIEW"):
        st.dataframe(df)

def st_display_sweetviz(report_html, width=1000, height = 500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)    
    
def show_shape():
    st.write(df.shape)   

def show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

def show_values(df3):
    missing_values_count = df3.isnull().sum()
    st.table(missing_values_count)  
    
def Descriptive_statistics():  
        val_description = df.describe()
        st.table(val_description)

df_clus = pd.read_csv("C:/Users/mahes/OneDrive/Desktop/BANK PROJECT/bank_clustering1.csv")
df_cluster=pd.DataFrame(df_clus)        
def observe_3d_relationships(variable_x, variable_y, variable_z, ax):

    ax.set_xlabel(variable_x.replace('_', ' '))
    ax.set_ylabel(variable_y.replace('_', ' '))
    ax.set_zlabel(variable_z.replace('_', ' '))

    x = df_cluster[variable_x].astype('float')
    y = df_cluster[variable_y].astype('float')
    z = df_cluster[variable_z].astype('float')

    ax.scatter(x, y, z)    
def explore_3d_relationship_of_variables(column_triplets):
    
    n = len(column_triplets)
    fig = plt.figure(figsize = (4 * n, 12))
    fig.subplots_adjust(hspace = 0.6, wspace = 0.6)
    
    for i in range(1, n + 1):
        column_triplet = column_triplets[i - 1]
        
        ax = fig.add_subplot(1, n, i, projection = '3d')
        observe_3d_relationships(column_triplet[0], column_triplet[1], column_triplet[2], ax)

    #plt.show()    
    st.pyplot()
    
def observe_relationships_with_scatterplot(variable_x, variable_y, ax):
    
    sns.scatterplot(data = df_cluster, x = variable_x, y = variable_y, ax = ax)
    
    plt.title(
        f'Relationship of variables:')
    plt.xlabel(variable_x.replace('_', ' '))
    plt.ylabel(variable_y.replace('_', ' ')) 

def explore_relationship_of_variables(column_pairs):
    n = len(column_pairs)

    fig = plt.figure(figsize = (5 * n, 3))
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
    
    for i in range(1, n + 1):
        
        column_pair = column_pairs[i - 1]

        ax = fig.add_subplot(1, n, i)
        observe_relationships_with_scatterplot(column_pair[0], column_pair[1], ax)

    #plt.show()  
    st.pyplot()     
                
def Clustered_Model():
        opt2 = ["Cluster1","Cluster2","Cluster3"]
        option2 = st.sidebar.selectbox("Select an option",opt2)
        df_clus = pd.read_csv("C:/Users/mahes/OneDrive/Desktop/BANK PROJECT/bank_clustering1.csv")
        df_cluster=pd.DataFrame(df_clus)  
        if option2 == "Cluster1":
                df_clus1 = df_cluster[df_cluster['cluster']== 0]
                data_view_func(df_clus1)
                st.subheader("3D View of 3 Variables of Cluster1")
                st.markdown(style, unsafe_allow_html=True)
                explore_3d_relationship_of_variables([
                ['Num_Bank_Accounts', 'Outstanding_Debt', 'Num_of_Loan'],
                ['Num_Credit_Inquiries', 'Delay_from_due_date', 'Num_of_Loan'],
                ['Num_of_Delayed_Payment', 'Interest_Rate', 'Num_of_Loan']])
                st.subheader(" Relationship of Variables in Cluster1")
                st.markdown(style, unsafe_allow_html=True)
                g = sns.PairGrid(df_clus1,vars=['Interest_Rate','Annual_Income','Num_of_Loan','Total_EMI_per_month','Monthly_Balance'])
                #setting color
                g.map_upper(sns.scatterplot, color='crimson')
                g.map_lower(sns.scatterplot, color='limegreen')
                g.map_diag(plt.hist, color='orange')
                #show figure
                st.pyplot()
                st.subheader("Number of Rows and Columns in Cluster1 ")
                st.markdown(style, unsafe_allow_html=True)
                st.write(df_clus1.shape)
        if option2 == "Cluster2":
                df_clus2 = df_cluster[df_cluster['cluster']== 1]  
                data_view_func(df_clus2) 
                st.subheader("3D View of 3 Variables of Cluster2")
                st.markdown(style, unsafe_allow_html=True)
                explore_3d_relationship_of_variables([
                ['Num_Bank_Accounts', 'Outstanding_Debt', 'Num_of_Loan'],
                ['Num_Credit_Inquiries', 'Delay_from_due_date', 'Num_of_Loan'],
                ['Num_of_Delayed_Payment', 'Interest_Rate', 'Num_of_Loan']]) 
                st.subheader(" Relationship of Variables in Cluster2")
                st.markdown(style, unsafe_allow_html=True)
                g = sns.PairGrid(df_clus2,vars=['Interest_Rate','Annual_Income','Num_of_Loan','Total_EMI_per_month','Monthly_Balance'])
                #setting color
                g.map_upper(sns.scatterplot, color='crimson')
                g.map_lower(sns.scatterplot, color='limegreen')
                g.map_diag(plt.hist, color='orange')
                #show figure
                st.pyplot()
                st.subheader("Number of Rows and Columns in Cluster2")
                st.markdown(style, unsafe_allow_html=True)
                st.write(df_clus2.shape)
        if option2 == "Cluster3":
                df_clus3 = df_cluster[df_cluster['cluster']== 2]   
                data_view_func(df_clus3)    
                st.subheader("3D View of 3 Variables of Cluster3")
                st.markdown(style, unsafe_allow_html=True)
                explore_3d_relationship_of_variables([
                ['Num_Bank_Accounts', 'Outstanding_Debt', 'Num_of_Loan'],
                ['Num_Credit_Inquiries', 'Delay_from_due_date', 'Num_of_Loan'],
                ['Num_of_Delayed_Payment', 'Interest_Rate', 'Num_of_Loan']]) 
                st.subheader(" Relationship of Variables in Cluster3")
                st.markdown(style, unsafe_allow_html=True)
                g = sns.PairGrid(df_clus3,vars=['Interest_Rate','Annual_Income','Num_of_Loan','Total_EMI_per_month','Monthly_Balance'])
                #setting color
                g.map_upper(sns.scatterplot, color='crimson')
                g.map_lower(sns.scatterplot, color='limegreen')
                g.map_diag(plt.hist, color='orange')
                #show figure
                st.pyplot()   
                st.subheader("Number of Rows and Columns in Cluster3") 
                st.markdown(style, unsafe_allow_html=True)
                st.write(df_clus3.shape)

def labeling(late_payment):
        '''
        we have late payments in range between 0 to 25, we will lable with below categorized value
        1. low risk: 0 to 2
        2. medium risk: 3 to 10
        3. high risk: 11 to 25
        
        '''
        if late_payment<=2:
            return 'low_risk'
        elif 3 <= late_payment <=10 :
            return 'medium_risk'
        else:
            return 'high_risk'
        
def loan(x,y):
        if ( x in ['Standard','Good'] ) and (y in ['low_risk','medium_risk']):
            return 'Approved'
        else:
            return 'Rejected'                 
         
                  
def classification_model():
        #Month_options = [1,2,3,4,5,6,7,8]
        occupation_options = ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer',
                                'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager',
                                'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']
        Num_bank_accounts = [1,2,3,4,5,6,7,8,9,10,11]
        intrest_rate = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
        num_of_loans = [0,1,2,3,4,5,6,7,8,9]
        num_of_delayed_payment = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        num_credit_enquires = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
        credit_mix = ['Good','Bad','Standard']
        payment_min_amount = ['Yes','No','NM']
        payment_behaviour = ['High_spent_Small_value_payments','Low_spent_Large_value_payments','Low_spent_Medium_value_payments',
                                'Low_spent_Small_value_payments','High_spent_Medium_value_payments','High_spent_Large_value_payments']
        age = [23, 28, 34, 54, 55, 21, 31, 33, 30, 24, 44, 45, 40, 41, 32, 35, 36,
                39, 37, 20, 46, 26, 42, 19, 48, 38, 43, 22, 16, 18, 15, 27, 25, 14,
                17, 47, 53, 56, 29, 49, 51, 50, 52]
        #type_of_loan = df['Type_of_Loan'].unique().tolist()
        #type_of_loan = ['Auto Loan', 'Credit-Builder Loan', 'Personal Loan', 'Home Equity Loan','Not Specified','Payday Loan','Student Loan','Mortgage Loan']
        # Define the widgets for user input
        with st.form("my_form"):
                col1, col2, col3 = st.columns([5, 2, 5])
                with col1:
                        st.write(' ')
                        Age = st.selectbox("Age", sorted(age), key=1)
                        Occupation = st.selectbox("Occupation", occupation_options, key=2)                      
                        Num_bank_acc = st.selectbox("Number of bank Accounts",Num_bank_accounts, key=3)
                        Num_credit_card = st.selectbox("Number of Credit Card", Num_bank_accounts, key=4)
                        Intrest_rate = st.selectbox("Interest Rate", intrest_rate, key=5)
                        Num_of_loans = st.selectbox("Number of loans", num_of_loans, key=6)
                        Num_of_Delayed_Payment = st.selectbox("Number of Delayed Payments", num_of_delayed_payment, key=7)
                        Num_Credit_Inquiries = st.selectbox("Number of Credit Enquiries", num_credit_enquires, key=8)
                        Credit_mix = st.selectbox("Credit Mix", credit_mix, key=9)
                        Payment_of_Min_Amount = st.selectbox("Payment of Minimum Amount", payment_min_amount, key=10)                       
                        Payment_Behaviour = st.selectbox("Payment Behaviour", payment_behaviour, key=11)
                        
                        
                with col3:
                        st.write(f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>',
                                unsafe_allow_html=True)
                        Name = st.text_input("Enter Name")
                        Annual_Income = st.text_input("Enter Annual_Income (Min:7005.93 & Max:179987.28)")
                        Monthly_Inhand_Salary = st.text_input("Enter Monthly_Inhand_Salary (Min:303.6454167 & Max:15204.63333)")
                        Delay_from_due_date = st.text_input("Delay_from_due_date (Min:0, Max:62)")
                        Changed_Credit_Limit = st.text_input("Changed_Credit_Limit (Min:0.5, Max:29.98)")
                        Outstanding_Debt = st.text_input("Outstanding_Debt (Min:0.23, Max:4998.07)")
                        Credit_Utilization_Ratio = st.text_input("Credit_Utilization_Ratio (Min:20.0, Max:50.0)")
                        Credit_History_Age = st.text_input("Credit_History_Age (Min:1, Max:404)")
                        Total_EMI_per_month = st.text_input("Total_EMI_per_month (Min:0.0, Max:1779.103254)")
                        Amount_invested_monthly = st.text_input("Amount_invested_monthly (Min:0.0, Max:434.1910894)")
                        Monthly_Balance = st.text_input("Monthly_Balance (Min:0.007759665, Max:1183.930696)")
                        submit_button = st.form_submit_button(label="PREDICT AVAILABILITY OF NEW CREDIT")
                        st.markdown("""
                                <style>
                                div.stButton > button:first-child {
                                        background-color: #009999;
                                        color: white;
                                        width: 100%;
                                }
                                </style>
                                """, unsafe_allow_html=True)

                        flag = 0
                        pattern = "^(?:\d+|\d*\.\d+)$"
                        for i in [Annual_Income,Monthly_Inhand_Salary,Delay_from_due_date,Changed_Credit_Limit,Outstanding_Debt,Credit_Utilization_Ratio,Credit_History_Age,Total_EMI_per_month,Amount_invested_monthly,Monthly_Balance]:
                                if re.match(pattern, i):
                                        pass
                                else:
                                        flag = 1
                                        break

                if submit_button and flag == 1:
                        if len(i) == 0:
                                st.write("please enter a valid number space not allowed")
                        else:
                                st.write("You have entered an invalid value: ", i)
                
                #encoding the categorigal columns
                #Credit_Mix_map = { 'Bad':0, 'Standard':1, 'Good': 2  }
                if Credit_mix == 'Bad':
                        Credit_mix = 0
                elif Credit_mix == 'Standard':
                        Credit_mix = 1
                elif Credit_mix == 'Good':
                        Credit_mix = 2
                                
                #Payment_of_Min_Amount_map = { 'No':0, 'NM':1, 'Yes':2 }
                if Payment_of_Min_Amount == 'No':
                        Payment_of_Min_Amount = 0
                elif Payment_of_Min_Amount == 'NM':
                        Payment_of_Min_Amount = 1
                elif Payment_of_Min_Amount == 'Yes':
                        Payment_of_Min_Amount = 2
                #Payment_Behaviour_map = { 'Low_spent_Small_value_payments': 0 , 'Low_spent_Medium_value_payments':1, 'Low_spent_Large_value_payments':2,
                                       # 'High_spent_Small_value_payments':3, 'High_spent_Medium_value_payments':4, 'High_spent_Large_value_payments':5
                                        #}
                
                if Payment_Behaviour == 'Low_spent_Small_value_payments':
                        Payment_Behaviour = 0
                elif Payment_Behaviour == 'Low_spent_Medium_value_payments':
                        Payment_Behaviour = 1
                elif Payment_Behaviour == 'Low_spent_Large_value_payments':
                        Payment_Behaviour = 2
                elif Payment_Behaviour == 'High_spent_Small_value_payments':
                        Payment_Behaviour = 3
                elif Payment_Behaviour == 'High_spent_Medium_value_payments':
                        Payment_Behaviour = 4
                elif Payment_Behaviour == 'High_spent_Large_value_payments':
                        Payment_Behaviour = 5
                        
                #Credit_mix = Credit_mix.map(Credit_Mix_map)
                
                #Payment_of_Min_Amount = Payment_of_Min_Amount.map(Payment_of_Min_Amount_map)
                
                #Payment_Behaviour = Payment_Behaviour.map(Payment_Behaviour_map)
                
                columns_data = [int(Age), int(Num_bank_acc), int(Num_credit_card), int(Num_of_loans), int(Delay_from_due_date), int(Num_of_Delayed_Payment),
                                int(Num_Credit_Inquiries), Credit_mix, int(Credit_History_Age), Payment_of_Min_Amount, Payment_Behaviour,
                                np.log10(float(Annual_Income)), np.log10(float(Monthly_Inhand_Salary)), np.log10(float(Intrest_rate)), np.log10(float(Changed_Credit_Limit)), np.log10(float(Outstanding_Debt)), 
                                np.log10(float(Credit_Utilization_Ratio)),np.log10(float(Total_EMI_per_month)), np.log10(float(Amount_invested_monthly)), np.log10(float(Monthly_Balance))]  
                #feature scaling
                scaling_model = load('bank_scaler.joblib')
                scaled_class = scaling_model.fit([columns_data]).transform([columns_data])
                
                #loading the trained model
                model = load('bank_classification.joblib')
                predicted = model.predict(scaled_class)  
                #Risk Assesment
                Risk_Assessment = labeling(Num_of_Delayed_Payment)

                # mapping the loan status based on the predicted value
                Loan_Status = loan(predicted,Risk_Assessment)
                
                if submit_button and flag == 0:
                        predicted_data = "Loan Availability Status: " + Loan_Status
                        st.success(predicted_data)
                
                        

                                        
        


####........ STREAMLIT CODING ........####    

if selected == "Home":
    html_temp = """
        <div style="background-color:#002e63;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">HOME</h3>
        </div>"""
    components.html(html_temp)    
    home_menu()   
  
if selected == "Data View and EDA":  
        html_temp = """
        <div style="background-color:#002e63;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">Data View and EDA</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
        components.html(html_temp)  
        choice = st.sidebar.selectbox("Choose an option",["Data View","Univariate Analysis","Bivariate Analysis","Multivariate Analysis","Wordcloud","Automated EDA"])
        if choice == "Data View":
                data_view_func(df)
                st.subheader("Number of rows and columns")
                st.markdown(style, unsafe_allow_html=True)
                show_shape()
                st.subheader("Information of dataset")
                st.markdown(style, unsafe_allow_html=True)
                s = show_info(df)
                st.text(s)
                st.subheader("Missing values count of each columns")
                st.markdown(style, unsafe_allow_html=True)
                show_values(df)
                st.subheader("Descriptive statistics")
                st.markdown(style, unsafe_allow_html=True)
                Descriptive_statistics()
        elif choice == "Univariate Analysis": 
                univariate_analysis(df)
        elif choice == "Bivariate Analysis":
                bivariate_analysis()
        elif choice == "Multivariate Analysis":   
                correlation_plot()        
        elif choice == "Wordcloud":
                wordcloud_fun(df)             
        elif choice == "Automated EDA":
                #profile = ProfileReport(data)   
                #st_profile_report(profile) 
                report = sv.analyze(df)
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")    
        else:
                pass    
if selected == "ML Prediction":
        html_temp = """
        <div style="background-color:#002e63;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">ML MODEL TRAINING AND PREDICTIONS</h3>
        </div>"""
        components.html(html_temp)   
        choice1 = st.sidebar.selectbox("Choose an option",["Clustered Model","Classification Model"]) 
        if choice1 == "Clustered Model":
                Clustered_Model()
        if choice1 == "Classification Model":
                classification_model()        
                       
                
            