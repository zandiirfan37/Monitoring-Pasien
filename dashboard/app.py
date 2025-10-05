from flask import Flask, render_template, request, jsonify
import requests
import json

app = Flask(__name__)

# Dummy patient data
patient_data = {
    "name": "John Doe",
    "age": 65,
    "room": 302,
    "picture": "https://via.placeholder.com/100"
}

@app.route('/')
def index():
    try:
        response = requests.get('http://127.0.0.1:5001/data')
        data = response.json()
    except requests.exceptions.ConnectionError:
        data = []
    return render_template('index.html', data=data, patient=patient_data)

@app.route('/data')
def get_data():
    try:
        response = requests.get('http://127.0.0.1:5001/data')
        data = response.json()
    except requests.exceptions.ConnectionError:
        data = []
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=5003)