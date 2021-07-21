# Aspect-Based Sentiment Analysis of Airline Tweets

The second classification task in this process is that of sentiment. Combined with the aspect of the tweet classified above, this yields aspect-based sentiment analysis to understand which airlines are receiving the most complaints against which facets of their customer service.

The baseline accuracy using a TF-IDF + Naive Bayes Classifier is: 70%.

An additional 4 sentiment models were evaluated against the Airline Tweets dataset:

VADER - SentimentIntensityAnalyzer (nltk): 65%
  Precision: 0.898
  Recall: 0.504
  Accuracy: 0.653
  F1 Score: 0.646
Textblob x NaiveBayesAnalyzer (nltk): 69%
  Precision: 0.775
  Recall: 0.716
  Accuracy: 0.692
  F1 Score: 0.744
Hugging Face (BERT): 79%
  Precision: 0.939
  Recall: 0.711
  Accuracy: 0.790
  F1 Score: 0.809
Fine-tuned Hugging Face (BERT): 89% on the test subset.
  With another airline tweets dataset:
  Precision: 0.853
  Recall: 0.738
  Accuracy: 0.791
  F1 Score: 0.791
Hugging Face with fine tuning was chosen as the final model.
