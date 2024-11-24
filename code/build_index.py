import pandas as pd
import nltk
import tkinter as tk
import os
from collections import defaultdict
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

df = pd.read_pickle("data/reviews_segment.pkl")
df["doc_id"] = range(1, len(df) + 1)
df.info(verbose=True)

# Preprocess Data
# Remove apostrophes from the data
df = df.applymap(lambda x: x.strip("'") if isinstance(x, str) else x)

# Build Inverted Index

# Initialize inverted index
inverted_index = defaultdict(set)
term_to_id = {}  # Maps terms to unique integer IDs
id_to_term = {}  # Reverse mapping of IDs to terms
current_id = 1  # Start term IDs from 1

stop_words = set(stopwords.words("english"))  # Load stopwords
lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer

print("Building Inverted Index...")

# Build the inverted index
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Inverted Index"):
    doc_id = row["doc_id"]
    words = nltk.word_tokenize(row['review_text'].lower())  # Tokenize and lowercase text
    
    for word in words:
        if word.isalpha() and word not in stop_words:  # Filter non-alphabetic and stop words
            word = lemmatizer.lemmatize(word)  # Lemmatize the word
            if word not in term_to_id:
                term_to_id[word] = current_id
                id_to_term[current_id] = word
                current_id += 1
            term_id = term_to_id[word]
            inverted_index[term_id].add(doc_id)  # Use .add() for set to avoid duplicates



# Convert sets to sorted lists for each term
inverted_index = {term_id: sorted(list(doc_ids)) for term_id, doc_ids in inverted_index.items()}

# Save mappings as pickle files
with open("index/term_to_id.pkl", "wb") as f:
    pickle.dump(term_to_id, f)
with open("index/id_to_term.pkl", "wb") as f:
    pickle.dump(id_to_term, f)

# Save the inverted index as a pickle file
with open("index/inverted_index.pkl", "wb") as f:
    pickle.dump(inverted_index, f)

# Also save CSV files for verification purposes
# Term-to-ID mapping
term_to_id_df = pd.DataFrame(list(term_to_id.items()), columns=["term", "term_id"])
term_to_id_df.to_csv("index/term_to_id.csv", index=False)

# ID-to-term mapping
id_to_term_df = pd.DataFrame(list(id_to_term.items()), columns=["term_id", "term"])
id_to_term_df.to_csv("index/id_to_term.csv", index=False)

# Inverted index CSV
inverted_index_rows = [
    {
        "term_id": term_id,
        "term": id_to_term[term_id],
        "postings_list": " → ".join(map(str, doc_ids))
    }
    for term_id, doc_ids in inverted_index.items()
]
inverted_index_df = pd.DataFrame(inverted_index_rows)
inverted_index_df.to_csv("index/inverted_index.csv", index=False)

print("All files have been saved: pickle files for submission and CSV files for verification.")

import sys

# Print size of the inverted index
print(f"Total unique terms in inverted index: {len(inverted_index)}")

# Print size of postings list for a sample term
sample_term_id = next(iter(inverted_index))  # Take the first term ID as a sample
print(f"Size of postings list for sample term ID {sample_term_id} ({id_to_term[sample_term_id]}): {len(inverted_index[sample_term_id])}")

# Print memory size of the entire inverted index in bytes and MB
memory_size_bytes = sys.getsizeof(inverted_index)
memory_size_mb = memory_size_bytes / (1024 ** 2)  # Convert bytes to MB
print(f"Memory size of inverted index: {memory_size_bytes} bytes ({memory_size_mb:.2f} MB)")