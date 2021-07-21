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

This is my revision of the great tutorial at http://brandonrose.org/clustering - many thanks to the author.

### TL;DR
**Data**: Top 100 movies (http://www.imdb.com/list/ls055592025/) with title, genre, and synopsis (IMDB and Wiki)

**Goal**: Put 100 movies into 5 clusters by text-mining their synopses and plot the result as follows

<img width="771" alt="screenshot 2016-05-23 20 50 20" src="https://butt.githubusercontent.com/assets/595772/15488829/5b863710-2128-11e6-843b-25aac76bd134.png">

### Setup

First, clone the repo, go to the repo folder, setup the virtual environment, and install the required packages:

```
$ cd path_to_document-clustering
$ virtualenv -p python3 venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```
Second, use nltk.download() to download all nltk packages (a GUI will open and you can choose to install all packages: ~ 3.5G), which are saved to /Users/your_mac_username/nltk_data

```
ipython
import nltk
nltk.download()
```

Lastly, run `$ jupyter notebook` to go over the tutorial step-by-step.

### Key Steps
1. **Read data**: read titles, genres, synopses, rankings into four arrays
2. **Tokenize and stem**: break paragraphs into sentences, then to words, stem the words (without removing stopwords) - each synopsis essentially becomes a bag of stemmed words.
3. **Generate tf-idf matrix**: each row is a term (unigram, bigram, trigram...generated from the bag of words in 2.), each column is a synopsis.
4. **Generate clusters**: based on the tf-idf matrix, 5 (or any number) clusters are generated using k-means. The top key terms are selected for each cluster.
5. **Calculate similarity**: generate the cosine similarity matrix using the tf-idf matrix (100x100), then generate the distance matrix (1 - similarity matrix), so each pair of synopsis has a distance number between 0 and 1.
6. **Plot clusters**: use multidimensional scaling (MDS) to convert distance matrix to a 2-dimensional array, each synopsis has (x, y) that represents their relative location based on the distance matrix. Plot the 100 points with their (x, y) using matplotlib (I added an example on using plotly.js).

## Part II. Sentiment Analysis

The second classification task in this process is that of sentiment. Combined with the aspect of the tweet classified above, this yields aspect-based sentiment analysis to understand which airlines are receiving the most complaints against which facets of their customer service.

The baseline accuracy using a TF-IDF + Naive Bayes Classifier is: 70%.

An additional 4 sentiment models were evaluated against the Airline Tweets dataset:

- VADER - SentimentIntensityAnalyzer (nltk): 65%
- Textblob x NaiveBayesAnalyzer (nltk): 69%
- Hugging Face (BERT): 79%
- Fine-tuned Hugging Face (BERT): 89%

Hugging Face with fine tuning was chosen as the final model.
