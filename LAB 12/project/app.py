from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model_SVR.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        num_courses = int(request.form['number_courses'])
        time_study = int(request.form['time_study'])

        input_data = np.array([[num_courses, time_study]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Predicted Marks: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
