from flask import Flask , render_template, jsonify
from flask import request

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 2: Explore the dataset (Optional)
# print(X.head())
# print(X.shape)
# print(y.value_counts())

#Step 3: Preprocess the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



# Step 5: Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/feature',methods=['POST'])
def feature_extraction():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    #  input_features =   [ 17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001,
    #         0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
    #         0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6,
    #         2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]

     if request.method == 'POST':
        # Get input features from the form
        input_features = [float(x) for x in request.form.values()]
    #     # Create a DataFrame with the input features
     feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                         'mean smoothness', 'mean compactness', 'mean concavity',
                         'mean concave points', 'mean symmetry', 'mean fractal dimension',
                         'radius error', 'texture error', 'perimeter error', 'area error',
                         'smoothness error', 'compactness error', 'concavity error',
                         'concave points error', 'symmetry error', 'fractal dimension error',
                         'worst radius', 'worst texture', 'worst perimeter', 'worst area',
                         'worst smoothness', 'worst compactness', 'worst concavity',
                         'worst concave points', 'worst symmetry', 'worst fractal dimension']
     df = pd.DataFrame([input_features], columns=feature_names)

        # Preprocess the input data (standardization)
    #  scaler = StandardScaler()
    #  input_scaled = scaler.fit_transform(df)
     input_scaled = scaler.transform(df)

        # Make predictions using the model
     prediction = model.predict(input_scaled)

        # Determine the prediction 
     result= "not sure"   
     if prediction[0] == 0:
            result = 'Benign'
     elif prediction[0] == 1:
            result = 'Malignant'
           
     return render_template('result.html',result = result)



if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)
