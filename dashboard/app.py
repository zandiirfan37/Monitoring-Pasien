
from flask import Flask, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    try:
        response = requests.get('http://127.0.0.1:5001/data')
        data = response.json()
    except requests.exceptions.ConnectionError:
        data = []
    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
