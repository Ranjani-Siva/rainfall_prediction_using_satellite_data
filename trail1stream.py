import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost import XGBClassifier
df=pd.read_csv("D:/miniprob24/processed_rainfall_dataset.csv")
df.head()
df.shape

z = np.abs(stats.zscore(df[['IMG_WV', 'IMG_MIR']]))
threshold = 3

df_filtered = df[(z < threshold).all(axis=1)]

print("Original DataFrame shape:", df.shape)
print("Filtered DataFrame shape:", df_filtered.shape)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df_filtered)
datas=pd.DataFrame(data_normalized)
datas.columns=df_filtered.columns
datas.head(5)

df_no_duplicates = datas.drop_duplicates()

print("Original DataFrame shape:", datas.shape)
print("DataFrame with duplicates removed:", df_no_duplicates.shape)
data=df_no_duplicates

# Define the independent variables (features) and dependent variable (target)
X = data[['IMG_MIR', 'IMG_SWIR', 'IMG_TIR1', 'IMG_TIR2', 'IMG_VIS', 'IMG_WV', 'Sat_Azimuth', 'Sat_Elevation', 'Sun_Azimuth', 'Sun_Elevation']]
y = data['Rainfall_Estimate']

# Add a constant to the model (intercept term)
X = sm.add_constant(X)

# Fit the OLS (Ordinary Least Squares) model
model = sm.OLS(y, X).fit()
# Get the summary of the model, including the p-values
summary = model.summary()
print(summary)

# Extract the p-values specifically
p_values = model.pvalues
print(p_values)

p_threshold = 0.05

# Perform backward elimination
while True:
    # Fit the model
    model = sm.OLS(y, X).fit()

    # Get p-values for all variables
    p_values = model.pvalues

    # Find the variable with the highest p-value
    max_p_value = p_values.max()

    # If the highest p-value is greater than the threshold, remove the variable
    if max_p_value > p_threshold:
        max_p_variable = p_values.idxmax()
        print(f"Removing {max_p_variable} with p-value {max_p_value}")
        X = X.drop(columns=[max_p_variable])  # Drop the variable with highest p-value
    else:
        break

# Display the final model summary
print(model.summary())


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    try:  # Handle cases where AUC-ROC might not be applicable
        y_prob = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_prob)
    except AttributeError:
        auc_roc = "Not applicable for this model"

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc}")
xgb_model = XGBClassifier(n_estimators=25, max_depth=10, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
print("\nXGBoost:")
evaluate_model(xgb_model, X_test, y_test)