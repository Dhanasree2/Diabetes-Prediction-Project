from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    output = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
    return render_template('index.html', prediction_text=f'The person is likely to be :{output}')

if __name__ == "__main__":
    app.run(debug=True)
