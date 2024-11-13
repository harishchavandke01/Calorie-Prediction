from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
# from tensorflow.keras.models import model_from_json
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from xgboost import DMatrix
import pandas as pd
# Load model architecture
# with open('model/xgb_reg.json', 'r') as json_file:
#     model_json = json_file.read()
# xgb_reg = model_from_json(model_json)

# Load model weights
# xgb_reg.load_weights('model/xgb_reg.h5')
model = xgb.Booster()
model.load_model('model/xgb_reg.json')


# Load scaler and label encoder
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
# with open('model/label_encoder.pkl', 'rb') as le_file:
#     label_encoder = pickle.load(le_file)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Female', 'Male'])

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()
    gender = data['Gender']
    age = data['Age']
    height = data['Height']
    weight = data['Weight']
    duration = data['Duration']
    heart_rate = data['Heart_Rate']
    body_temp = data['Body_Temp']
    
    # Process inputs
    gender_encoded = label_encoder.transform([gender])[0]
    # features = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])
    features = pd.DataFrame([{
        'Gender': gender_encoded,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Duration': duration,
        'Heart_Rate': heart_rate,
        'Body_Temp': body_temp
    }])
    features = features[['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']]
    features_scaled = scaler.transform(features)
    

    features_dmatrix = DMatrix(features_scaled, feature_names=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    predicted_calories = model.predict(features_dmatrix)[0]
    # print(features.columns)
    predicted_calories = float(predicted_calories)
    
    return jsonify({'calories': predicted_calories})

if __name__ == '__main__':
    app.run(debug=True)
