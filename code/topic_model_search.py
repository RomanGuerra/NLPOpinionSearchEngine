import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm
import pickle
import re

# Load the dataset
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Directory to save LDA outputs
output_dir = "../lda"

# Tokenization and Vectorization
vectorizer = CountVectorizer(stop_words="english", max_features=5000)
doc_term_matrix = vectorizer.fit_transform(df["review_text_cleaned"].fillna(""))

# Load data
doc_topic_distributions = pd.read_pickle(f"{output_dir}/document_topic_distribution.pkl")
with open(f"{output_dir}/lda_model.pkl", "rb") as f:
    lda_model = pickle.load(f)
with open(f"{output_dir}/vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)

print("\033[32mLDA artifacts loaded successfully.\033[0m")


tfidf_vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(df["review_text_cleaned"].fillna(""))

def match_query_to_topics(query, vectorizer, lda_model, vocabulary):
    """
    Match a query to the most relevant topics using the LDA model.
    """
    query_vector = vectorizer.transform([query])
    query_topic_distribution = lda_model.transform(query_vector).flatten()
    return query_topic_distribution

def retrieve_documents_by_topics(query_topic_distribution, doc_topic_distributions, threshold=0.1):
    """
    Retrieve documents based on their topic distribution similarity to the query.
    """
    matching_docs = []
    for doc_idx, doc_distribution in enumerate(doc_topic_distributions):
        similarity = np.dot(query_topic_distribution, doc_distribution)
        if similarity >= threshold:  # Adjust threshold as needed
            matching_docs.append(doc_idx)
    return matching_docs


def topic_model_search(aspect1, aspect2, opinion, method, similarity_threshold=0.35):
    """
    Perform topic modeling search with filtering for relevance.

    Args:
        aspect1 (str): First aspect of the query.
        aspect2 (str): Second aspect of the query.
        opinion (str): Opinion or sentiment in the query.
        method (str): Boolean method ("method1", "method2", "method3").
        similarity_threshold (float): Minimum similarity score to consider a document relevant.

    Returns:
        DataFrame: Filtered results ranked by topic similarity.
    """
    # Combine query terms into a single query string
    query = f"{aspect1} {aspect2} {opinion}"
    
    # Match query to topics
    query_topic_distribution = match_query_to_topics(query, vectorizer, lda_model, vocabulary)
    
    # Retrieve relevant documents
    matching_doc_ids = retrieve_documents_by_topics(query_topic_distribution, doc_topic_distributions, threshold=similarity_threshold)
    
    # Filter DataFrame to include only matching documents
    results_df = df.iloc[matching_doc_ids].copy()
    
    # Add topic similarity scores for ranking
    results_df["topic_similarity"] = [
        np.dot(query_topic_distribution, doc_distribution)
        for doc_distribution in doc_topic_distributions[matching_doc_ids]
    ]
    
    # Further filter results by similarity threshold
    results_df = results_df[results_df["topic_similarity"] >= similarity_threshold]
    
    # Sort results by topic similarity
    results_df = results_df.sort_values(by="topic_similarity", ascending=False)
    
    return results_df
