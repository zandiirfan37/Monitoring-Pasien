from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import os

app = Flask(__name__)
CORS(app)
DB_FILE = os.path.join(os.path.dirname(__file__), 'database.db')
# The script directory is one level above the backend directory
STATIC_FOLDER = os.path.join(os.path.dirname(__file__), '..')

def init_db():
    if not os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE patient_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                audio_path TEXT,
                prediction TEXT
            )
        """)
        conn.commit()
        conn.close()

@app.route('/data', methods=['POST'])
def receive_data():
    data = request.get_json()
    image_path = data.get('image_path')
    audio_path = data.get('audio_path')
    prediction = data.get('prediction')

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO patient_data (image_path, audio_path, prediction) VALUES (?, ?, ?)",
              (image_path, audio_path, prediction))
    conn.commit()
    conn.close()

    return jsonify({'message': 'Data received successfully'}), 201

@app.route('/data', methods=['GET'])
def get_data():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM patient_data ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    conn.close()

    # Convert to a list of dictionaries and construct URLs
    data = []
    for row in rows:
        data.append({
            'id': row['id'],
            'timestamp': row['timestamp'],
            'image_url': f'/files/{os.path.basename(row["image_path"])}' if row["image_path"] else None,
            'audio_url': f'/files/{os.path.basename(row["audio_path"])}' if row["audio_path"] else None,
            'prediction': row['prediction']
        })

    return jsonify(data)

@app.route('/files/<path:filename>')
def serve_file(filename):
    # Determine the subdirectory (images or audios) based on the file extension
    if filename.endswith('.jpg'):
        directory = os.path.join(STATIC_FOLDER, 'images')
    elif filename.endswith('.wav'):
        directory = os.path.join(STATIC_FOLDER, 'audios')
    else:
        return "File not found", 404
    return send_from_directory(directory, filename)

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5001)