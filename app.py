from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load the model
with open('dtr.pkl', 'rb') as f:
    dtr = pickle.load(f)

# Recreate the preprocessor (to avoid version compatibility issues)
ohe = OneHotEncoder(drop='first')
scaler = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('onehotencoder', ohe, ['Item', 'Area']),
        ('standardscaler', scaler, [
         'Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'])
    ],
    remainder='passthrough'
)

# Load the data to fit the preprocessor
df = pd.read_csv('yield_df.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.drop_duplicates(inplace=True)

# Prepare features and fit the preprocessor
X = df.drop(columns=['hg/ha_yield'])
y = df['hg/ha_yield']

# Fit the preprocessor on the full dataset
preprocessor.fit(X)


def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create a DataFrame with the same column names and order as training data
    features_df = pd.DataFrame({
        'Year': [Year],
        'average_rain_fall_mm_per_year': [average_rain_fall_mm_per_year],
        'pesticides_tonnes': [pesticides_tonnes],
        'avg_temp': [avg_temp],
        'Area': [Area],
        'Item': [Item]
    })

    # Transform the features using the preprocessor
    transformed_features = preprocessor.transform(features_df)

    # Make the prediction
    predicted_yield = dtr.predict(transformed_features)

    return predicted_yield[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the form data
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(
            request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Make the prediction
        result = prediction(Year, average_rain_fall_mm_per_year,
                            pesticides_tonnes, avg_temp, Area, Item)

        return render_template('index.html', prediction=result)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
