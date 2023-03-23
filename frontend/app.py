from flask import Flask, render_template, request, jsonify
from emotion_extraction import emotion_extraction

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/emotion_extraction', methods=['POST'])
def extract_emotion():
    video_url = request.form['video_url']
    result = emotion_extraction(video_url)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
