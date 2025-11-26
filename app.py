import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import time

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.set_page_config(page_title="ML Classifier Comparison",page_icon="web_icon.png")

st.title("Automated ML Classifier Comparison App")
st.write("by Sakib Hossain Tahmid")


st.markdown(
    """
    <style>
    .round-img {
        width: 40px;
        height: 35px;
        border-radius: 50%;
        object-fit: cover;
        transition: 0.2s;
    }
    .round-img:hover {
        transform: scale(1.1);
    }
    </style>
    <span style="color:blue;">>>></span>
     <span style="color:orange;">Try our No-Code ML App .Build custom ML models without coding and Download it!</span>
    <a href="https://nocodemlsakib.streamlit.app/" target="_blank">
        <img src="https://github.com/sakib-12345/No-Code-ML-WEBapp/blob/main/my_icon.png?raw=true" class="round-img">
    </a>  
    
    """,
    unsafe_allow_html=True
)



uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        with st.expander("Preview Data"):
            st.dataframe(df)
            st.write("Instructions:")
        with st.expander("Summary Statistics"):
            st.write(df.describe())
        with st.expander("Column Data Types"):
            st.write(df.dtypes)
            
        feature = st.multiselect("Select Training Columns(must be numerical):", df.columns.tolist() if uploaded_file is not None else ["no option"])
        target = st.selectbox("Select Target Column(must be 0 or 1):", df.columns.tolist() if uploaded_file is not None else ["none"])

        if uploaded_file is not None and df.isnull().sum().sum() > 0:
            fill_value = st.selectbox("Fill Null Value With:", ["Mean","Median","Drop"] if uploaded_file is not None else ["no value"])
            miss_val = df.isnull().sum().sum()
            st.markdown(f'Total <span style="color:red;">&nbsp;{miss_val}&nbsp;</span>missing values in dataset', unsafe_allow_html=True)
            with st.expander("Missing Values"):
                st.write(df.isnull().sum())
        else:
            st.success("No missing values in dataset ")
            fill_value = None
        start = st.button("Train Model")    
    except Exception:
        df = pd.DataFrame()
        st.error("File loading error. Please upload a valid CSV file.")
        start = False
else:
    st.write("> Please upload a CSV file to get started.")        
if uploaded_file is not None and feature and target and start:  
    try:    
        X = df[feature]
        y = df[target]

        if fill_value == "Mean":
            X = X.fillna(X.mean())
        elif fill_value == "Median":
            X = X.fillna(X.median())
        elif fill_value == "Drop":
            X = X.dropna()
            y = y[X.index]
          

#standard scaling 
        rfc_std = RandomForestClassifier()
        lr_std = LogisticRegression()
        dtc_std = DecisionTreeClassifier()
        knn_std = KNeighborsClassifier()
        svc_std = SVC()

        scaler_std = StandardScaler()
        X_std = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns)
        X_std_train, X_std_test, y_std_train, y_std_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

        rfc_std.fit(X_std_train, y_std_train)
        lr_std.fit(X_std_train, y_std_train)
        dtc_std.fit(X_std_train, y_std_train)
        knn_std.fit(X_std_train, y_std_train)
        svc_std.fit(X_std_train, y_std_train)

        y_std_rfc = rfc_std.predict(X_std_test)
        y_std_lr = lr_std.predict(X_std_test)
        y_std_dtc = dtc_std.predict(X_std_test)
        y_std_knn = knn_std.predict(X_std_test)
        y_std_svc = svc_std.predict(X_std_test)

        score_std_rfc = accuracy_score(y_std_test, y_std_rfc)*100
        score_std_lr = accuracy_score(y_std_test, y_std_lr)*100
        score_std_dtc = accuracy_score(y_std_test, y_std_dtc)*100
        score_std_knn = accuracy_score(y_std_test, y_std_knn)*100
        score_std_svc = accuracy_score(y_std_test, y_std_svc)*100


#minmax scaling
        rfc_mms = RandomForestClassifier()
        lr_mms = LogisticRegression()
        dtc_mms = DecisionTreeClassifier()
        knn_mms = KNeighborsClassifier()
        svc_mms = SVC()  

        scaler_mms = MinMaxScaler()
        X_mms= pd.DataFrame(scaler_mms.fit_transform(X), columns=X.columns)
        X_mms_train, X_mms_test, y_mms_train, y_mms_test = train_test_split(X_mms, y, test_size=0.2, random_state=42)

        rfc_mms.fit(X_mms_train, y_mms_train)
        lr_mms.fit(X_mms_train, y_mms_train)
        dtc_mms.fit(X_mms_train, y_mms_train)
        knn_mms.fit(X_mms_train, y_mms_train)
        svc_mms.fit(X_mms_train, y_mms_train)

        y_mms_rfc = rfc_mms.predict(X_mms_test)
        y_mms_lr = lr_mms.predict(X_mms_test)
        y_mms_dtc = dtc_mms.predict(X_mms_test)
        y_mms_knn = knn_mms.predict(X_mms_test)
        y_mms_svc = svc_mms.predict(X_mms_test)

        score_mms_rfc = accuracy_score(y_mms_test, y_mms_rfc)*100
        score_mms_lr = accuracy_score(y_mms_test, y_mms_lr)*100
        score_mms_dtc = accuracy_score(y_mms_test, y_mms_dtc)*100
        score_mms_knn = accuracy_score(y_mms_test, y_mms_knn)*100
        score_mms_svc = accuracy_score(y_mms_test, y_mms_svc)*100
        
        data = {
            "Random Forest Classifier(Standard Scaling)": score_std_rfc,
            "Logistic Regression(Standard Scaling)": score_std_lr,
            "Decision Tree Classifier(Standard Scaling)": score_std_dtc,
            "K-Nearest Neighbors(Standard Scaling)": score_std_knn,
            "Support Vector Classifier(Standard Scaling)": score_std_svc,

            "Random Forest Classifier(MinMax Scaling)": score_mms_rfc,
            "Logistic Regression(MinMax Scaling)": score_mms_lr,
            "Decision Tree Classifier(MinMax Scaling)": score_mms_dtc,
            "K-Nearest Neighbors(MinMax Scaling)": score_mms_knn,
            "Support Vector Classifier(MinMax Scaling)": score_mms_svc
        }
        result_df = pd.DataFrame.from_dict(data,orient='index', columns=['Accuracy'])
        df_sorted = result_df.sort_values(by='Accuracy', ascending=False)
        st.markdown("### Results Summary:(top 3 models)")
        st.dataframe(df_sorted.head(3))
        top_model_name = df_sorted.index[0]
        low_model_name = df_sorted.index[-1]
        st.markdown("---")
        st.markdown(f'**Top Accuracy Model:**&nbsp;&nbsp;{top_model_name}<span style="color:green;">&nbsp;&nbsp;**↑**</span>',unsafe_allow_html=True)
        st.markdown(f'**Low Accuracy Model:**&nbsp;&nbsp;{low_model_name}<span style="color:red;">&nbsp;&nbsp;**↓**</span>',unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("## Model Performance:")
     
        with st.expander("Random Forest Classifier"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Standard Scaling Results:")
                st.write("Accuracy:", round(score_std_rfc, 2),"%")
                st.write("Precision:", round(precision_score(y_std_test, y_std_rfc)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_std_test, y_std_rfc)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_std_test, y_std_rfc)*100, 2),"%")
            with col2:
                st.markdown("##### MinMax Scaling Results:")
                st.write("Accuracy:", round(score_mms_rfc, 2),"%")
                st.write("Precision:", round(precision_score(y_mms_test, y_mms_rfc)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_mms_test, y_mms_rfc)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_mms_test, y_mms_rfc)*100, 2),"%")
        with st.expander("Logistic Regression"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Standard Scaling Results:")
                st.write("Accuracy:", round(score_std_lr, 2),"%")
                st.write("Precision:", round(precision_score(y_std_test, y_std_lr)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_std_test, y_std_lr)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_std_test, y_std_lr)*100, 2),"%")
            with col2:
                st.markdown("##### MinMax Scaling Results:")
                st.write("Accuracy:", round(score_mms_lr, 2),"%")
                st.write("Precision:", round(precision_score(y_mms_test, y_mms_lr)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_mms_test, y_mms_lr)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_mms_test, y_mms_lr)*100, 2),"%")
        with st.expander("Decision Tree Classifier"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Standard Scaling Results:")
                st.write("Accuracy:", round(score_std_dtc, 2),"%")
                st.write("Precision:", round(precision_score(y_std_test, y_std_dtc)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_std_test, y_std_dtc)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_std_test, y_std_dtc)*100, 2),"%")
            with col2:
                st.markdown("##### MinMax Scaling Results:")
                st.write("Accuracy:", round(score_mms_dtc, 2),"%")
                st.write("Precision:", round(precision_score(y_mms_test, y_mms_dtc)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_mms_test, y_mms_dtc)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_mms_test, y_mms_dtc)*100, 2),"%")
        with st.expander("K-Nearest Neighbors"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Standard Scaling Results:")
                st.write("Accuracy:", round(score_std_knn, 2),"%")
                st.write("Precision:", round(precision_score(y_std_test, y_std_knn)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_std_test, y_std_knn)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_std_test, y_std_knn)*100, 2),"%")
            with col2:
                st.markdown("##### MinMax Scaling Results:")
                st.write("Accuracy:", round(score_mms_knn, 2),"%")
                st.write("Precision:", round(precision_score(y_mms_test, y_mms_knn)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_mms_test, y_mms_knn)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_mms_test, y_mms_knn)*100, 2),"%")
        with st.expander("Support Vector Classifier"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Standard Scaling Results:")
                st.write("Accuracy:", round(score_std_svc, 2),"%")
                st.write("Precision:", round(precision_score(y_std_test, y_std_svc)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_std_test, y_std_svc)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_std_test, y_std_svc)*100, 2),"%")
            with col2:
                st.markdown("##### MinMax Scaling Results:")
                st.write("Accuracy:", round(score_mms_svc, 2),"%")
                st.write("Precision:", round(precision_score(y_mms_test, y_mms_svc)*100, 2),"%")
                st.write("Recall:", round(recall_score(y_mms_test, y_mms_svc)*100, 2),"%")
                st.write("F1 Score:", round(f1_score(y_mms_test, y_mms_svc)*100, 2),"%")                               
    except Exception:
        st.error("An error occurred during model training. Please ensure that the selected features are numerical and the target column is binary (0 or 1).")
        



st.markdown(
    """
    <style>
        .social-icons {
            text-align: center;
            margin-top: 60px;
        }

        .social-icons a {
            text-decoration: none !important;
            margin: 0 20px;
            font-size: 28px;
            display: inline-block;
            color: inherit !important; /* force child i to use its color */
        }

        

        /* Hover glitch animation */
        .social-icons a:hover {
            animation: glitch 0.3s infinite;
        }

        
        /* Contact us heading */
        .contact-heading {
            text-align: center;
            font-size: 25px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        @keyframes glitch {
            0% { transform: translate(0px, 0px); text-shadow: 2px 2px #0ff, -2px -2px #f0f; }
            20% { transform: translate(-2px, 1px); text-shadow: -2px 2px #0ff, 2px -2px #f0f; }
            40% { transform: translate(2px, -1px); text-shadow: 2px -2px #0ff, -2px 2px #f0f; }
            60% { transform: translate(-1px, 2px); text-shadow: -2px 2px #0ff, 2px -2px #f0f; }
            80% { transform: translate(1px, -2px); text-shadow: 2px -2px #0ff, -2px 2px #f0f; }
            100% { transform: translate(0px, 0px); text-shadow: 2px 2px #0ff, -2px -2px #f0f; }
        }
    </style>
    <div class="social-icons">
    <div class="contact-heading">Contact Us:</div>
        <a class='fb' href='https://www.facebook.com/sakibhossain.tahmid' target='_blank'>
            <i class='fab fa-facebook'></i> 
        </a> 
        <a class='insta' href='https://www.instagram.com/_sakib_000001' target='_blank'>
            <i class='fab fa-instagram'></i> 
        </a> 
        <a class='github' href='https://github.com/sakib-12345' target='_blank'>
            <i class='fab fa-github'></i> 
        </a> 
        <a class='email' href='mailto:sakibhossaintahmid@gmail.com'>
            <i class='fas fa-envelope'></i> 
        </a>
    </div>
    
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """,
    unsafe_allow_html=True
)


st.markdown(
            f'<div style="text-align: center; color: grey;">&copy; 2025 Sakib Hossain Tahmid. All Rights Reserved.</div>',
            unsafe_allow_html=True
           ) 




