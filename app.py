import streamlit as st
import pandas as pd
import joblib

# 1. Load the trained model
model = joblib.load("beer_prediction_model.pkl")

st.title("🍺 Beer Servings Prediction App")

# 2. Get User Input
country = st.text_input("Country", "United States")
beer = st.number_input("Beer Servings", min_value=0, max_value=500, value=100)
spirit = st.number_input("Spirit Servings", min_value=0, max_value=500, value=50)
wine = st.number_input("Wine Servings", min_value=0, max_value=500, value=50)

continent = st.selectbox(
    "Continent",
    ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]
)

# 3. Handle One-Hot Encoding matching your model training
# Initialize all continent columns as False
continent_data = {
    'continent_Asia': False,
    'continent_Europe': False,
    'continent_North America': False,
    'continent_Oceania': False,
    'continent_South America': False
}

# Set the selected continent column to True (except Africa, which was dropped as first)
if f"continent_{continent}" in continent_data:
    continent_data[f"continent_{continent}"] = True

# 4. Create DataFrame for Prediction
input_data = pd.DataFrame([{
    "beer_servings": beer,
    "spirit_servings": spirit,
    "wine_servings": wine,
    **continent_data
}])

# 5. Predict Button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(
        f"Predicted Total Litres of Pure Alcohol: {prediction[0]:.2f}"
    )