import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pickle
import os
from boolean_search import boolean_search

# Load the dataset
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Directory to save the TF-IDF artifacts
output_dir = "../scoring"

df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Load TF-IDF vectorizer and matrix
with open("../scoring/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
with open("../scoring/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)
print("TF-IDF vectorizer and matrix loaded.")

# Verify that loaded objects are correct
assert hasattr(tfidf_vectorizer, "transform"), "tfidf_vectorizer is not a TfidfVectorizer object."
assert hasattr(tfidf_matrix, "shape"), "tfidf_matrix is not a valid sparse matrix."

def compute_similarity(query, tfidf_vectorizer, tfidf_matrix):
    """
    Compute semantic similarity between the query and the TF-IDF matrix.
    
    Args:
        query (str): Query string.
        tfidf_vectorizer (TfidfVectorizer): Pretrained TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): Precomputed TF-IDF matrix.
        
    Returns:
        list: Similarity scores for each document in the TF-IDF matrix.
    """
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    return similarity_scores


def boolean_semantic_search(aspect1, aspect2, opinion, method):
    """
    Perform Boolean search and rank results by semantic similarity.
    
    Args:
        aspect1, aspect2, opinion (str): Query terms.
        method (str): Boolean search method ("method1", "method2", "method3").
        
    Returns:
        DataFrame: Results ranked by semantic similarity.
    """
    # Load the TF-IDF vectorizer and matrix
    output_dir = "../scoring"
    with open(f"{output_dir}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(f"{output_dir}/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    # Verify the types of loaded objects
    assert hasattr(tfidf_vectorizer, "transform"), "tfidf_vectorizer is not a TfidfVectorizer object."
    assert hasattr(tfidf_matrix, "shape"), "tfidf_matrix is not a valid sparse matrix."

    # Perform Boolean search to get matching docs
    matching_docs = boolean_search(aspect1, aspect2, opinion, method)
    
    # Filter DataFrame to matching docs
    matched_df = df[df["doc_id"].isin(matching_docs)].copy()
    
    # Combine query terms into a single string
    query = f"{aspect1} {aspect2} {opinion}"
    
    # Compute similarity scores for matched docs
    query_vector = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Add similarity scores to the DataFrame
    matched_df["semantic_score"] = matched_df.index.map(lambda idx: similarity_scores[idx])
    
    # Filter by similarity threshold
    similarity_threshold=0.2
    matched_df = matched_df[matched_df["semantic_score"] >= similarity_threshold]

    # Sort by semantic score in descending order
    matched_df = matched_df.sort_values(by="semantic_score", ascending=False)
    
    return matched_df
