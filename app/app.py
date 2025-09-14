from flask import Flask, render_template, request
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts import process


app = Flask(__name__)


model = joblib.load("outputs/model.pkl")
vectorizer = joblib.load("outputs/vectorizer.pkl")


positive_words = {"great", "amazing", "best", "happy", "love", "awesome", "fantastic"}
negative_words = {"stfu", "hate", "worst", "terrible", "sucks", "angry", "mad"}

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        raw_text = request.form["tweet"]
        text = process.clean_tweet(raw_text)


        if text.strip() == "":
            sentiment = "Neutral"
            return render_template("home.html", prediction_text=f"Sentiment: {sentiment}", tweet=raw_text)


        text_vec = vectorizer.transform([text])
        neg_prob, pos_prob = model.predict_proba(text_vec)[0]

        if text_vec.nnz == 0:
            sentiment = "Neutral"


        elif abs(pos_prob - neg_prob) < 0.2:
            sentiment = "Neutral"

       
        elif any(word in text.split() for word in positive_words):
            sentiment = "Positive"
        elif any(word in text.split() for word in negative_words):
            sentiment = "Negative"

        elif pos_prob > neg_prob:
            sentiment = "Positive"
        else:
            sentiment = "Negative"

        return render_template("home.html", prediction_text=f"Sentiment: {sentiment}", tweet=raw_text)

if __name__ == '__main__':
    app.run(debug=True)
