
import pandas as pd
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import os
import re

from boolean_search import boolean_search

print("\033[33mRating Search...\033[0m")

# Importing the data
nltk_data_path = os.path.join(os.getcwd(), '../nltk_data')
nltk.data.path.append(nltk_data_path)

# Load Data
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

def load_opinion_lexicon():
    positive_words_file = "../opinion-lexicon-English/positive-words.txt"
    negative_words_file = "../opinion-lexicon-English/negative-words.txt"

    with open(positive_words_file, "r", encoding="latin-1") as f:
        positive_words = set(line.strip() for line in f if not line.startswith(";") and line.strip())

    with open(negative_words_file, "r", encoding="latin-1") as f:
        negative_words = set(line.strip() for line in f if not line.startswith(";") and line.strip())

    return positive_words, negative_words

# Load the lexicon
positive_words, negative_words = load_opinion_lexicon()

def determine_opinion_sentiment(opinion):

    opinion_words = opinion.lower().split()
    if any(word in positive_words for word in opinion_words):
        return "Positive"
    elif any(word in negative_words for word in opinion_words):
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["review_text_cleaned"].apply(determine_opinion_sentiment)

def boolean_rating_search(aspect1, aspect2, opinion, method):

    # Perform the Boolean search to get matching document IDs (this will be based on your previous implementation)
    matching_docs = boolean_search(aspect1, aspect2, opinion, method)

    # Get matched reviews
    matched_df = df[df['doc_id'].isin(matching_docs)]

    # Determine sentiment based on the query's opinion term
    sentiment_label = determine_opinion_sentiment(opinion)

    # Apply filtering based on sentiment and rating
    if sentiment_label == "Positive":
        filtered_df = matched_df[
            (matched_df["customer_review_rating"] > 3) & 
            (matched_df["sentiment_label"] == "Positive")
        ]
    elif sentiment_label == "Negative":
        filtered_df = matched_df[
            (matched_df["customer_review_rating"] <= 3) & 
            (matched_df["sentiment_label"] == "Negative")
        ]
    else:  # If sentiment is neutral or unknown, return all matching docs without rating filtering
        filtered_df = matched_df[
            (matched_df["customer_review_rating"] >= 1)
        ]
    # print(filtered_df.head())
    return filtered_df