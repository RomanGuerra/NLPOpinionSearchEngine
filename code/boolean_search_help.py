import pandas as pd
import argparse
import pickle
import argparse
from boolean_search import boolean_search
from boolean_rating_search import boolean_rating_search
from boolean_semantic_search import boolean_semantic_search
from topic_model_search import topic_model_search
from pathlib import Path

# Load inverted index and term mappings
with open("../index/inverted_index.pkl", "rb") as f:
    inversed_index = pickle.load(f)

with open("../index/term_to_id.pkl", "rb") as f:
    term_to_id = pickle.load(f)

def method1(aspect1, aspect2, opinion):
    """
    the first method 
    """
    return boolean_search(aspect1, aspect2, opinion, method="method2")

def method2(aspect1, aspect2, opinion):
    """
    the second method 
    """
    return boolean_rating_search(aspect1, aspect2, opinion, method="method2")

def method3(aspect1, aspect2, opinion):
    """
    the third method 
    """
    return boolean_semantic_search(aspect1, aspect2, opinion, method="method2")

def method4(aspect1, aspect2, opinion):
    """
    the fourth method 
    """
    return topic_model_search(aspect1, aspect2, opinion, method="method2")

def save_results(result, aspect1, aspect2, opinion, method_name):
    """
    Format and save the results to a pickle file with specified naming convention.
    """
    # Ensure unique document IDs and format output as required
    unique_results = list(set(result))
    revs = pd.DataFrame(unique_results, columns=["review_id"])

    # Define output path
    filename = f"../output/{method_name}_{aspect1}_{aspect2}_{opinion}.pkl"
    revs.to_pickle(filename)
    print(f"Results saved to {filename}")

def main():

    parser = argparse.ArgumentParser(description="Perform the boolean search.")
    
    parser.add_argument("-a1", "--aspect1", type=str, required=True, default=None, help="First word of the aspect")
    parser.add_argument("-a2", "--aspect2", type=str, required=True, default=None, help="Second word of the aspect")
    parser.add_argument("-o", "--opinion", type=str, required=True, default=None, help="Only word of the opinion")
    parser.add_argument("-m", "--method", type=str, required=True, default=None, help="The method of boolean operation. Methods\
                        can be method1, method2 or method3")

    # Parse the arguments
    args = parser.parse_args()

    review_df = pd.read_pickle("../data/reviews_segment_processed.pkl")

    aspect1, aspect2, opinion = args.aspect1, args.aspect2, args.opinion

    if args.method.lower() == "method1":
        result = method1(aspect1, aspect2, opinion)
        # Map doc_id to review_id
        result_review_ids = review_df.loc[review_df['doc_id'].isin(result), 'review_id']
        revs = pd.DataFrame({"review_id": result_review_ids})
        print(revs)

    elif args.method.lower() == "method2":
        result = method2(aspect1, aspect2, opinion)

        if isinstance(result, pd.DataFrame):  # If already a DataFrame (as returned by `boolean_rating_search`)
            if 'review_id' in result.columns:
                revs = result[['review_id']].copy()
            else:
                print("\n!! No 'review_id' column found in the DataFrame for Method 2 !!\n")
                return
        elif isinstance(result, set):  # If the result is a set of `doc_id`s
            result_review_ids = review_df.loc[review_df['doc_id'].isin(result), 'review_id']
            revs = pd.DataFrame({"review_id": result_review_ids})
        else:  # Handle unexpected result types
            print("\n!! Unexpected result type for Method 2 !!\n")
            return

        print(revs)

    elif args.method.lower() == "method3":
        result = method3(aspect1, aspect2, opinion)
        if isinstance(result, pd.DataFrame):  # If already a DataFrame (as returned by `boolean_rating_search`)
            if 'review_id' in result.columns:
                revs = result[['review_id']].copy()
            else:
                print("\n!! No 'review_id' column found in the DataFrame for Method 2 !!\n")
                return
        elif isinstance(result, set):  # If the result is a set of `doc_id`s
            result_review_ids = review_df.loc[review_df['doc_id'].isin(result), 'review_id']
            revs = pd.DataFrame({"review_id": result_review_ids})
        else:  # Handle unexpected result types
            print("\n!! Unexpected result type for Method 2 !!\n")
            return
    elif args.method.lower() == "method4":
        result = method1(aspect1, aspect2, opinion)
        # Map doc_id to review_id
        result_review_ids = review_df.loc[review_df['doc_id'].isin(result), 'review_id']
        revs = pd.DataFrame({"review_id": result_review_ids})
        print(revs)
    else:
        print("\n!! The method is not supported !!\n")
        return

    # Save results
    filename = f"../output/{args.method}_{args.aspect1}_{args.aspect2}_{args.opinion}.pkl"
    revs.to_pickle(filename)

if __name__ == "__main__":
    main()
