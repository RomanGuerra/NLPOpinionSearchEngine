# NLPOpinionSearchEngine
 An advanced natural language opinion search engine leveraging NLP techniques, aspect-based sentiment analysis, and Boolean search logic to enhance query precision and relevance, showcasing expertise in algorithm development and search optimization for real-world applications.


### Environment Setup
This project is using Anaconda to install libraries.

To recreate the environment needed for this project, follow the steps below:

1. Install **Conda** if not already installed.
   
2. Create the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

### Project Structure
- **build_index.py**: Script to preprocess data and build the inverted index.
- **boolean_search.py**: Script containing the main Boolean search function and helper functions.
- **boolean_search_help.py**: Script for running and testing different query methods.
- **data/**: Folder to store pickle data files.
- **output/**: Folder to store generated pickle files for each query and method.
- **index/**: Folder containing serialized index files and CSV files.
- **nltk_data/**: You will need to download NLTK libraries to this folder.

### Building Index
Not needed. Index is already built, so this command is here just incase.
```bash
python build_index.py
```
### Running the Search
You can run the Boolean search using the provided `boolean_search.py` script. Here's an example:

```bash
python boolean_search_help.py --aspect1 "audio" --aspect2 "quality" --opinion "poor" --method "method1"
```

### Available Query Methods
1. **Method 1**: aspect1 OR aspect2 OR opinion
2. **Method 2**: aspect1 AND aspect2 AND opinion
3. **Method 3**: (aspect1 OR aspect2) AND opinion

### Running Project Landmark

Not needed. Copy this entire command into your terminal to create output files for project landmark.
```
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method1
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method2
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method3
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method1
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method2
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method3
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method1
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method2
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method3
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method1
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method2
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method3
```