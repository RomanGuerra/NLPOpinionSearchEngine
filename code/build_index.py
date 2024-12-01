import pandas as pd
import os
from collections import defaultdict
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from tqdm import tqdm
import re
import sys

print("\033[33mBuilding Inverted Index...\033[0m")

# Importing the data
nltk_data_path = os.path.join(os.getcwd(), '../nltk_data')
nltk.data.path.append(nltk_data_path)

# Read data
df = pd.read_pickle("../data/reviews_segment_processed.pkl")

# Paths for output files
output_dir = "../index"
os.makedirs(output_dir, exist_ok=True)

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Initialize components
postings_list = []
term_to_id = {}
id_to_term = {}
term_freq = defaultdict(int)
current_id = 1

# Additional statistics
stats = {
    "total_words": 0,
    "alphabetic_words": 0,
    "numeric_words": 0,
    "stopwords": 0,
    "lemmatized_words": 0
}

print("\033[33m")
# Process reviews to build postings list and term mappings
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Postings List"):
    doc_id = row["doc_id"]
    words = nltk.word_tokenize(row["review_text_cleaned"].lower())

    for word in words:
        stats["total_words"] += 1
        if word.isalpha():
            stats["alphabetic_words"] += 1

        if word.isdigit():
            stats["numeric_words"] += 1
            
        if word in stop_words:
            stats["stopwords"] += 1

        if word.isalpha() and word not in stop_words:
            lemma = lemmatizer.lemmatize(word)
            stats["lemmatized_words"] += 1
            if lemma not in term_to_id:
                term_to_id[lemma] = current_id
                id_to_term[current_id] = lemma
                current_id += 1
            term_id = term_to_id[lemma]
            postings_list.append((term_id, doc_id))
            term_freq[term_id] += 1
print("\033[0m")

print("\033[33mSorting Postings List...\033[0m")

# Sort the postings list
postings_list = sorted(postings_list, key=lambda x: (x[0], x[1]))

# Postings List with Terms (Sorted by Term and Document ID)
postings_list_terms = [
    {"Term": id_to_term[term_id], "Document ID": doc_id}
    for term_id, doc_id in postings_list
]

# Sort the postings_list_terms explicitly by Term and Document ID before saving
postings_list_terms_df = pd.DataFrame(postings_list_terms)
postings_list_terms_df = postings_list_terms_df.sort_values(by=["Term", "Document ID"], ascending=[True, True])

print("\033[33mBuilding Inverted Index...\033[0m")

# Build the inverted index from the sorted postings list
inverted_index = defaultdict(list)
for term_id, doc_id in postings_list:
    if not inverted_index[term_id] or inverted_index[term_id][-1] != doc_id:
        inverted_index[term_id].append(doc_id)

# Save Pickle Files
with open(f"{output_dir}/inverted_index.pkl", "wb") as f:
    pickle.dump(dict(inverted_index), f)
with open(f"{output_dir}/term_to_id.pkl", "wb") as f:
    pickle.dump(term_to_id, f)
with open(f"{output_dir}/id_to_term.pkl", "wb") as f:
    pickle.dump(id_to_term, f)
with open(f"{output_dir}/postings_list.pkl", "wb") as f:
    pickle.dump(postings_list, f)
with open(f"{output_dir}/term_freq.pkl", "wb") as f:
    pickle.dump(term_freq, f)
with open(f"{output_dir}/stats.pkl", "wb") as f:
    pickle.dump(stats, f)

# Save Statistics
pd.DataFrame(stats.items(), columns=["Metric", "Count"]).to_csv(f"{output_dir}/stats.csv", index=False)

# Print Statistics
print("\033[36m")
print(f"Total unique terms in inverted index: {len(inverted_index):,}")
memory_size_bytes = sys.getsizeof(inverted_index)
memory_size_mb = memory_size_bytes / (1024 ** 2)
print(f"Memory size of inverted index: {memory_size_bytes:,} bytes ({memory_size_mb:.2f} MB)")

print("\nTotal words:", stats["total_words"])
print("Alphabetic words:", stats["alphabetic_words"])
print("Numeric words:", stats["numeric_words"])
print("Stopwords:", stats["stopwords"])
print("Lemmatized words:", stats["lemmatized_words"])
print("\033[0m")

print("\033[32mBuilding Inverted Index Complete.\033[0m")