from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model and LabelEncoder
model = pickle.load(open('iri.pkl', 'rb'))
le = LabelEncoder()
le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])  # Ensure that the encoder is fitted to your class labels

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Getting data from the form
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    
    # Converting form data to a numpy array
    arr = np.array([[data1, data2, data3, data4]])
    
    # Make prediction
    pred = model.predict(arr)
    
    # Decode the prediction if needed
    pred_label = le.inverse_transform(pred)  # This converts numerical predictions back to original class names
    
    # Print to debug prediction
    print(f"Prediction: {pred}")
    print(f"Decoded Prediction: {pred_label[0]}")
    
    return render_template('after.html', data=pred_label[0])

if __name__ == "__main__":
    app.run(debug=True)











