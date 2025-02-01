
import pandas as pd
import joblib
from sklearn.metrics import r2_score as R2Score

# This needs to be changed with a method that pulls data from a source (Bucket, DB, DWH, etc)
# Using a static name, can make the pulled content dynamic and always output different predictions.
df = pd.read_csv("./datasets/blind_test_data.csv")

# This needs to be changed with a method that loads the model from a source (Repo, Bucket, linrary, etc)
# models should not be hardcoded, but loaded dynamically when available
model = joblib.load("./model.pkl")

df_original = pd.read_csv('./datasets/training_data.csv') # This is used to scale data back to original values
df_original_no_target = df_original.drop('target', axis=1)

if __name__ == '__main__':
    normalized_df=(df-df_original_no_target.min())/(df_original_no_target.max()-df_original_no_target.min())
    prediction = model.predict(normalized_df)
    normalized_df['target'] = prediction
    denormalized_df = normalized_df * ( df_original.max() - df_original.min() ) + df_original.min()
    print(denormalized_df['target'])
    # This needs to be changed with a method that pushes data to a destination (Bucket, DB, DWH, etc)
    # Here we can also add monitoring of the model, like logging the prediction, the model used, the time it took to make the prediction, etc.
    denormalized_df.to_csv("./datasets/prediction.csv", index=False)
