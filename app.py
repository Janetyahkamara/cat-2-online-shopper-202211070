from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_model.pkl")

@app.route('/')
def home():
    return render_template("form.html")

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # Read the input values
        input_data = [float(x) for x in request.form.values()]
        # Make prediction
        prediction = model.predict([np.array(input_data)])
        result = "✅ Will Purchase" if prediction[0] == 1 else "❌ Will Not Purchase"
        return render_template("form.html", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
