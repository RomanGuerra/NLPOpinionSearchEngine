import pandas as pd
import pickle
from collections import defaultdict
import os
import nltk
from nltk.stem import WordNetLemmatizer

print("\033[33mBoolean Search...\033[0m")

# Importing the data
nltk_data_path = os.path.join(os.getcwd(), '../nltk_data')
nltk.data.path.append(nltk_data_path)

# Load the document data (adjust the path as needed)
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Load inverted index from Pickle file (optimized loading)
with open("../index/inverted_index.pkl", "rb") as f:
    inversed_index = pickle.load(f)
with open("../index/term_to_id.pkl", "rb") as f:
    term_to_id = pickle.load(f)

lemmatizer = WordNetLemmatizer()
def lemmatize_terms(terms):
    return [lemmatizer.lemmatize(term) for term in terms]

# Boolean search helper functions
def intersect(p1, p2):
    answer = []
    i, j = 0, 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]:
            answer.append(p1[i])
            i += 1
            j += 1
        elif p1[i] < p2[j]:
            i += 1
        else:
            j += 1
    return answer

def union(p1, p2):
    return sorted(set(p1).union(set(p2)))

def boolean_search(aspect1, aspect2, opinion, method):
    min_length = 2
    # Filter terms based on minimum length
    aspect1 = aspect1 if len(aspect1) >= min_length else None
    aspect2 = aspect2 if len(aspect2) >= min_length else None
    opinion = opinion if len(opinion) >= min_length else None

    # Remove None values before searching
    terms = [t for t in [aspect1, aspect2, opinion] if t]

    # Lemmatize query terms
    aspect1, aspect2, opinion = lemmatize_terms([aspect1, aspect2, opinion])

    # Convert terms to their respective term IDs
    aspect1_id = term_to_id.get(aspect1.lower())
    aspect2_id = term_to_id.get(aspect2.lower())
    opinion_id = term_to_id.get(opinion.lower())

    # Retrieve document IDs from the inverted index
    aspect1_docs = set(inversed_index.get(aspect1_id, []))
    aspect2_docs = set(inversed_index.get(aspect2_id, []))
    opinion_docs = set(inversed_index.get(opinion_id, []))

    # Perform Boolean operations based on the selected method
    if method == "method1":
        # OR operation on everything
        result_docs = aspect1_docs.union(aspect2_docs).union(opinion_docs)
    elif method == "method2":
        # AND operation on everything
        result_docs = aspect1_docs.intersection(aspect2_docs).intersection(opinion_docs)
    elif method == "method3":
        # OR on aspects, AND with opinion
        aspect_docs = aspect1_docs.union(aspect2_docs)
        result_docs = aspect_docs.intersection(opinion_docs)
    else:
        print("Invalid method.")
        return []        
    return result_docs