import pandas as pd
import pickle
from collections import defaultdict
import os
import nltk
from nltk.stem import WordNetLemmatizer

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

lemmatizer = WordNetLemmatizer()

# Load Reviews Segment
df = pd.read_pickle("data/reviews_segment.pkl")
df["doc_id"] = range(1, len(df) + 1)

# Load Inverted Index
with open("index/inverted_index.pkl", "rb") as f:
    inversed_index = pickle.load(f)

with open("index/term_to_id.pkl", "rb") as f:
    term_to_id = pickle.load(f)

# Intersection Function
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

# Union Function
def union(p1, p2):
    return sorted(set(p1).union(set(p2)))

# Boolean Search Function
def boolean_search(aspect1, aspect2, opinion, method):
    # Lemmatize the query terms
    aspect1 = lemmatizer.lemmatize(aspect1.lower())
    aspect2 = lemmatizer.lemmatize(aspect2.lower())
    opinion = lemmatizer.lemmatize(opinion.lower())

    # Convert terms to their respective term IDs
    aspect1_id = term_to_id.get(aspect1)
    aspect2_id = term_to_id.get(aspect2)
    opinion_id = term_to_id.get(opinion)

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

    # Map `doc_id` results to `review_id`
    review_ids = df[df["doc_id"].isin(result_docs)]["review_id"].tolist()
    
    return review_ids