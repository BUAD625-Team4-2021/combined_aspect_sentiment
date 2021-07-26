## We are Team Hugging Face <img src="40-408954_smile-face-hugging-emoji-png-download-clipart-emoji.png" width="100" />

# Aspect-Based Sentiment Analysis of Airline Tweets

## Project Summary

This project examines tweets and identifies the the aspect or topic of that tweet as well as the customer’s sentiment towards it. This can be used to give a company an evaluation of the topic and its severity, so they can task out whether it requires remediation. The project applies this approach to tweets directed to airlines about their customer service.

## Outcomes

- Fine-tuning the Hugging Face model for sentiment analysis showed drastic improvements in sentiment detection comparatively to the baseline. If the model we’re to deteriorate with time, we are confident that minor tweaks could keep its current accuracy. 
- Aspect clustering was not as straight forward and required a lot of manual intervention and as of now would not be sustainable. This portion could potentially be outsourced.
- When customer service is using our model they can continually add input whether a tweet is correctly classified, adding to our training data over time
- Lastly, we would love to expand our original training dataset to a more current timeframe. With Covid-19 changing travel so drastically there may be new things that aren’t covered with our training set.

## Recommendation

With a refined solution for the aspect identification, our work serves as the basis for building a social listening tool to:
- Inform customer service on customer complaints are trending
- Discover new pockets of negative sentiment, or new topics (e.g. complaints surrounding the use of COVID masks)
- Benchmarking against other companies (i.e. where do we perform better, what should we advertise?)

# Analysis Overview
See below for an overview of each component of the project, aspect identification and sentiment analysis.

## Part I. Document Clustering with Python 3

When you have downloaded all the tweets, use ntlk.download() to get packages for language processing such as stop words and stemming.

### Key Steps
1. **Read data**: read tweets and clean them of all hashtags, mentions, and URLs
2. **Tokenize and stem**: break tweets into sentences, then to words, stem the words (without removing stopwords) - each synopsis essentially becomes a bag of stemmed words.
3. **Generate tf-idf matrix**: each row is a term (unigram, bigram, trigram...generated from the bag of words in 2.), each column is a synopsis. Use 20% of the words in tweets, because most words are just repeats or don't provide much value. 
4. **Generate clusters**: based on the tf-idf matrix, 8 clusters are generated using k-means. The top key terms are selected for each cluster.
5. **Human Interpretation**: Break these clusters into 5 well defined aspects and create a dictionary that will assign aspects to a tweet based on the words found in it.

## Part II. Sentiment Analysis

The second classification task in this process is that of sentiment. Combined with the aspect of the tweet classified above, this yields aspect-based sentiment analysis to understand which airlines are receiving the most complaints against which facets of their customer service.

The baseline accuracy using a TF-IDF + Naive Bayes Classifier is: 70%.

An additional 4 sentiment models were evaluated against the Airline Tweets dataset:

### Key Models
- VADER - SentimentIntensityAnalyzer (nltk): 65%
- Textblob x NaiveBayesAnalyzer (nltk): 69%
- Hugging Face (BERT): 79%
- Fine-tuned Hugging Face (BERT): 89%

Hugging Face with fine tuning was chosen as the final model.
