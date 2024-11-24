import pandas as pd
import argparse
import pickle
import argparse
from boolean_search import boolean_search  # Import boolean_search function
from pathlib import Path

# Load inverted index and term mappings
with open("index/inverted_index.pkl", "rb") as f:
    inversed_index = pickle.load(f)

with open("index/term_to_id.pkl", "rb") as f:
    term_to_id = pickle.load(f)

def method1(aspect1, aspect2, opinion):
    """
    the first method will only perform the aspect1 OR aspect2 OR opinion
    """
    return boolean_search(aspect1, aspect2, opinion, method="method1")

def method2(aspect1, aspect2, opinion):
    """
    the second method will only perform the aspect1 AND aspect2 AND opinion
    """
    return boolean_search(aspect1, aspect2, opinion, method="method2")

def method3(aspect1, aspect2, opinion):
    """
    the third method will only perform the aspect1 OR aspect2 AND opinion
    """
    return boolean_search(aspect1, aspect2, opinion, method="method3")

def save_results(result, aspect1, aspect2, opinion, method_name):
    """
    Format and save the results to a pickle file with specified naming convention.
    """
    # Ensure unique document IDs and format output as required
    unique_results = list(set(result))
    revs = pd.DataFrame(unique_results, columns=["review_id"])

    # Define output path
    filename = f"output/{method_name}_{aspect1}_{aspect2}_{opinion}.pkl"
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

    review_df = pd.read_pickle("data/reviews_segment.pkl")

    aspect1, aspect2, opinion = args.aspect1, args.aspect2, args.opinion

    if args.method.lower() == "method1":
        result = method1(aspect1, aspect2, opinion)
    elif args.method.lower() == "method2":
        result = method2(aspect1, aspect2, opinion)
    elif args.method.lower() == "method3":
        result = method3(aspect1, aspect2, opinion)
    else:
        print("\n!! The method is not supported !!\n")
        return

    # revs = pd.DataFrame()
    # revs["review_index"] = [r[1:-1] for r in result] #making sure, I am not having the quotes on the index
    # revs.to_pickle(args.aspect1 + "_" + args.aspect2 + "_" + args.opinion + "_" + args.method + ".pkl")

    revs = pd.DataFrame({"review_id": list(result)})
    filename = f"output/{args.method}_{args.aspect1}_{args.aspect2}_{args.opinion}.pkl"
    revs.to_pickle(filename)

if __name__ == "__main__":
    main()
