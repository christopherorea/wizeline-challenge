
import pandas as pd
import joblib
from sklearn.metrics import r2_score as R2Score
import streamlit as st

@st.cache_resource
def get_model():
    # This needs to be changed with a method that loads the model from a source (Repo, Bucket, linrary, etc)
    # models should not be hardcoded, but loaded dynamically when available
    return joblib.load("./model.pkl")

st.title('Prediction POC - Christopher Orea')
uploaded_file = st.file_uploader("Upload a CSV file. 200 Mbs Max.", type=["csv"])

df_original = pd.read_csv('./datasets/training_data.csv') # This is used to scale data back to original values
df_original_no_target = df_original.drop('target', axis=1)

if uploaded_file is not None:
    model = get_model()

    df = pd.read_csv(uploaded_file)
    normalized_df=(df-df_original_no_target.min())/(df_original_no_target.max()-df_original_no_target.min())
    prediction = model.predict(normalized_df)
    normalized_df['target'] = prediction
    denormalized_df = normalized_df * ( df_original.max() - df_original.min() ) + df_original.min()
    st.write(denormalized_df)

st.markdown(f"[Let's start working together](https://www.linkedin.com/in/chrisgalleta/)") 
