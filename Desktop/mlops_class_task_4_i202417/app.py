from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load("wine_quality_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({'quality': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
