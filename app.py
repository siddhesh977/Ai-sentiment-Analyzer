
from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


app = Flask(__name__)


analyzer = SentimentIntensityAnalyzer()
try:
    analyzer.polarity_scores('test') 
except LookupError:
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def home():
    """Renders the main page with the sentiment form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyzes the submitted text and displays the result."""
    text = request.form['text_input']
    scores = analyzer.polarity_scores(text)

    compound_score = scores['compound']
    sentiment = ""

    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return render_template('result.html', text=text, scores=scores, sentiment=sentiment)

if __name__ == '__main__':
    
    app.run(debug=True)