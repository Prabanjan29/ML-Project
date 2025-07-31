import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize Flask App
flask_app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load Pickle Model
model = pickle.load(open("C:\\Users\\Prabanjan\\OneDrive\\Desktop\\Project 1\\Crop Recommendation\\Crop Recommendation\\model.pkl", "rb"))

# Home Route
@flask_app.route("/")
def Home():
    return render_template("index.html")

# Prediction Route
@flask_app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text=f"The Predicted Crop is {prediction[0]}")

# Run App
if __name__ == "__main__":
    flask_app.run(debug=True)
