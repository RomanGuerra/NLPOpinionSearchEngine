import pandas as pd
import re

print("\033[32mPreprocessing the data...\033[0m")

# Read data
df = pd.read_pickle("../data/reviews_segment.pkl")

# Create the 'doc_id' column
df["doc_id"] = range(1, len(df) + 1)

# Reorder the columns to make 'doc_id' the first column
cols = ['doc_id'] + [col for col in df.columns if col != 'doc_id']
df = df[cols]

# Remove apostrophes from the data
df = df.map(lambda x: x.strip("'") if isinstance(x, str) else x)

# Convert numerical strings to numeric types
numerical_cols = ['helpful_count', 'out_of_helpful_count', 'customer_review_rating', 'number_of_comments', 'amazon_verified_purchase']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert review_written_date to datetime
df['review_written_date'] = pd.to_datetime(df['review_written_date'], errors='coerce')

def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Replace escaped newline characters (\n\n and \n) with a space
    text = re.sub(r'\\n+', ' ', text)
    
    # Remove backslashes used for escaping (e.g., \')
    text = text.replace("\\'", "'").replace('\\"', '"').replace("\\", "")
    
    # Replace special dashes with a standard dash
    text = re.sub(r'[–—]', '-', text)
    
    # Replace multiple periods with a single ellipsis
    text = re.sub(r'\.\.\.+', '…', text)
    
    # Remove excessive punctuation (e.g., !!!, ???)
    text = re.sub(r'([!?])\1+', r'\1', text)
    
    # Remove non-essential symbols except for currency, sentence end, hyphens, ellipses
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\-$€£…]", '', text)
    
    # Standardize single and double quotes
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    
    # Normalize spaces around punctuation
    text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Apply the cleaning function to the 'review_text' column
df['review_text_cleaned'] = df['review_text'].apply(clean_text)

# Display the cleaned text
print("\nCleaned text:")
print("\033[33m")
pd.set_option('display.max_colwidth', 100)
print(df['review_text_cleaned'].head())
print("\033[0m")

# Save the cleaned DataFrame to a pickle file
df.to_pickle("../data/reviews_segment_processed.pkl")
print("\nDataFrame saved to ../data/reviews_segment_processed.pkl")

print("\n\033[32mPreprocessing complete.\033[0m")