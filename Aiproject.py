from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')


analyzer = SentimentIntensityAnalyzer()


sentence1 = "This movie was absolutely fantastic and I loved every minute of it!"
sentence2 = "The service was not good, and the food was terrible."
sentence3 = "I'm not sure how I feel about this."

print(analyzer.polarity_scores(sentence1))
print(analyzer.polarity_scores(sentence2))
print(analyzer.polarity_scores(sentence3))