
import pandas as pd
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import os
import re

from boolean_rating_search import load_opinion_lexicon, determine_opinion_sentiment

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

#############################################
##### Rating Search Preprocessing ###########
print("\033[33mRating Search...\033[0m")

# Importing the data
nltk_data_path = os.path.join(os.getcwd(), '../nltk_data')
nltk.data.path.append(nltk_data_path)

# Load Data
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Load the lexicon
positive_words, negative_words = load_opinion_lexicon()

df["sentiment_label"] = df["review_text_cleaned"].apply(determine_opinion_sentiment)

# Save the cleaned DataFrame to a pickle file
df.to_pickle("../data/reviews_segment_processed.pkl")
print("\nDataFrame saved to ../data/reviews_segment_processed.pkl")

print("\n\033[32mPreprocessing complete.\033[0m")

#############################################
##### Semantic Search Preprocessing #########

# Load the dataset
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Directory to save the TF-IDF artifacts
output_dir = "../scoring"
os.makedirs(output_dir, exist_ok=True)

# Create TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["review_text_cleaned"].fillna(""))

# Save TF-IDF vectorizer and matrix
with open(f"{output_dir}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)
with open(f"{output_dir}/tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)
print("TF-IDF files saved.")

# Print basic info about the TF-IDF matrix
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Number of terms: {len(tfidf_vectorizer.get_feature_names_out())}")

#############################################
##### Topic Model Search Preprocessing ##########

# Load the dataset
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Directory to save LDA outputs
output_dir = "../lda"
os.makedirs(output_dir, exist_ok=True)

# Tokenization and Vectorization
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
doc_term_matrix = vectorizer.fit_transform(df["review_text_cleaned"].fillna(""))

# Save vocabulary
with open(f"{output_dir}/vocabulary.pkl", "wb") as f:
    pickle.dump(vectorizer.get_feature_names_out(), f)

# Train LDA model
num_topics = 50
lda_model = LatentDirichletAllocation(
    n_components=num_topics,
    random_state=42,
    max_iter=20,
    learning_method="online",
    batch_size=128,
)
lda_model.fit(doc_term_matrix)

# Save LDA model
with open(f"{output_dir}/lda_model.pkl", "wb") as f:
    pickle.dump(lda_model, f)

def display_topics(model, feature_names, no_top_words):
    """
    Display the top words in each topic from the LDA model.
    """
    topics_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics_dict[topic_idx] = top_features
    return topics_dict

# Extract topics
vocabulary = vectorizer.get_feature_names_out()
topics = display_topics(lda_model, vocabulary, 10)

# Save topics to a file
with open(f"{output_dir}/topics.pkl", "wb") as f:
    pickle.dump(topics, f)

# Print topics for verification
print("Extracted Topics:")
for topic_id, words in topics.items():
    print(f"Topic {topic_id}: {', '.join(words)}")


# Get per-document topic distributions
doc_topic_distributions = lda_model.transform(doc_term_matrix)

# Assign most probable topic to each document
df["dominant_topic"] = np.argmax(doc_topic_distributions, axis=1)
df["topic_distribution"] = doc_topic_distributions.tolist()

# Save the document-topic distribution
df.to_pickle(f"{output_dir}/document_topic_distribution.pkl")

