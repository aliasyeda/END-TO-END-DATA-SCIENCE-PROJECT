from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/')
def home():
    return "✅ Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        news_text = data['text']

        # Transform the text using the saved vectorizer
        text_vector = vectorizer.transform([news_text])

        # Make a prediction
        prediction = model.predict(text_vector)

        result = "Real News" if prediction[0] == 1 else "Fake News"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
