
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score as R2Score
from tensorflow.random import set_seed

set_seed(42) # Set seed for reproducibility

df = pd.read_csv("./datasets/blind_test_data.csv")

pca = PCA(n_components=10)
pca.fit(df)

pca_df = pd.DataFrame(
    pca.transform(df),
    columns=[f"component_{i}" for i in range(0, 10)]
)

model = load_model('model.keras')

if __name__ == '__main__':
    prediction = model.predict(pca_df)
    print(prediction)
    pd.DataFrame(prediction).to_csv("./datasets/prediction.csv", index=False)
