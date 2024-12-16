import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import streamlit.components.v1 as components
from xgboost import XGBClassifier
import requests
import joblib

st.title("Prediction of heavy /high impact rain events using satellite data")

st.header("Data Source")
st.subheader("Rainfall Dataset from satellite")
df=pd.read_csv("D:/miniprob24/processed_rainfall_dataset.csv")
with st.expander("Click Here"):
    st.dataframe(df.sample(100))

st.subheader("Gather input values from satellite")
locations = ["Coimbatore", "Mumbai", "Delhi", "Chennai", "Bangalore"]
city = st.selectbox("Select a location:", locations)
def get_current_water_vapor():
    # Replace with your OpenWeatherMap API key
    api_key = "d089c45704efe220f04d53385a670af1"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        humidity = data['main']['humidity']  
        temperature = data['main']['temp']  
        # Approximate water vapor content calculation
        # Using: Absolute Humidity (g/m³) = 6.112 × exp((17.67 × T) / (T + 243.5)) × H × 2.1674 / (273.15 + T)
        from math import exp
        absolute_humidity = (
            6.112 * exp((17.67 * temperature) / (temperature + 243.5)) * humidity * 2.1674 / (273.15 + temperature)
        )
        return round(absolute_humidity, 2)  # g/m³
    else:
        return f"Error fetching data: {response.status_code}"

water_vapor = get_current_water_vapor()
st.write("Current Water Vapor in ",city, ": ",water_vapor," g/m³")

z = np.abs(stats.zscore(df[['IMG_WV', 'IMG_MIR']]))
threshold = 3

df_filtered = df[(z < threshold).all(axis=1)]

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df_filtered)
datas=pd.DataFrame(data_normalized)
datas.columns=df_filtered.columns


# Define the independent variables (features) and dependent variable (target)
X_train = datas[['IMG_WV']]
y_train = datas['Rainfall_Estimate']

xgb_model = XGBClassifier(n_estimators=25, max_depth=10, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

model_data = joblib.load('D:/miniprob24/xgb_model_with_threshold.pkl')
xgb_model = model_data['model']
optimal_threshold = model_data['optimal_threshold']

if water_vapor:
    # Example real-time data input, replace with live data if available
    input_data = {
        "IMG_WV": water_vapor
    }
    input_df = pd.DataFrame([input_data])
    y_pred_proba = xgb_model.predict_proba(input_df)[:, 1]
    prediction = (y_pred_proba >= optimal_threshold).astype(int)
    if(prediction[0]==0):
        st.write("Predicted Rainfall Estimate: No Rainfall")
    else:
        st.write("Predicted Rainfall Estimate: High Rainfall")
else:
    st.error("Failed to fetch current water vapor data.")