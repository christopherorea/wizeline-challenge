
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score as R2Score
from tensorflow.random import set_seed
import streamlit as st

@st.cache_resource
def get_model():
    # This needs to be changed with a method that loads the model from a source (Repo, Bucket, linrary, etc)
    # models should not be hardcoded, but loaded dynamically when available
    return load_model('model.keras')

st.title('Prediction POC - Christopher Orea')
uploaded_file = st.file_uploader("Upload a CSV file. 200 Mbs Max.", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    pca = PCA(n_components=10)
    pca.fit(df)

    pca_df = pd.DataFrame(
        pca.transform(df),
        columns=[f"component_{i}" for i in range(0, 10)]
    )
    model = get_model()
    prediction = model.predict(pca_df)
    df['prediction'] = prediction
    st.write(df)

st.markdown(f"[Let's start working together](https://www.linkedin.com/in/chrisgalleta/)") 
