import json
import joblib   # use joblib instead of pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model correctly
regmodel = joblib.load("simple_linear_model.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    req_data = request.get_json()
    data = req_data["data"]

    new_data = np.array(list(data.values())).reshape(1, -1)
    prediction = regmodel.predict(new_data)

    return jsonify({
        "prediction": float(np.array(prediction).flatten()[0])
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    print(final_input)
    output = regmodel.predict(final_input)
    # Use .item() to extract scalar from NumPy array
    return render_template("home.html", prediction_text=f"The Performance Index prediction is {output.item():.2f}")


if __name__ == "__main__":
    app.run(debug=True)
