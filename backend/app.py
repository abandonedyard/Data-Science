from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///comments.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Определение модели данных для комментариев
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(200), nullable=False)
    polarity = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# Загрузка лексикона VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

@app.route('/')
def hello_world():
    return jsonify(message="Hello, World!")

@app.route('/comments', methods=['GET', 'POST'])
def handle_comments():
    if request.method == 'POST':
        content = request.json.get('comment')
        if not content:
            return jsonify({'error': 'Bad Request', 'message': 'No comment provided'}), 400

        scores = sia.polarity_scores(content)
        polarity = scores['compound']  # Используем compound score

        comment = Comment(text=content, polarity=polarity)
        db.session.add(comment)
        db.session.commit()
        return jsonify({'id': comment.id, 'text': comment.text, 'polarity': polarity}), 201

    elif request.method == 'GET':
        comments = Comment.query.all()
        return jsonify([{'id': comment.id, 'text': comment.text, 'polarity': comment.polarity, 'created_at': comment.created_at.isoformat()} for comment in comments])

# Создание таблиц и запуск приложения
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Создание таблиц базы данных
    app.run(debug=True)
