# NLPOpinionSearchEngine
 An advanced natural language opinion search engine leveraging NLP techniques, aspect-based sentiment analysis, and Boolean search logic to enhance query precision and relevance, showcasing expertise in algorithm development and search optimization for real-world applications.


### Environment Setup
This project is using Anaconda to install libraries.

To recreate the environment needed for this project, follow the steps below:

1. Install **Conda** if not already installed.

2. Change to **code/** directory.
   ```bash
   cd code
   ```

3. Create the Conda **NLP** environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

### Project Structure
- **code/ :** Folder to store code.
   - **preprocess.py**: Script to to preprocess data.
   - **build_index.py**: Script build the inverted index.
   - **boolean_search.py**: Script containing the main Boolean search function.
   - **boolean_search_help.py**: Script for running and testing different query methods.

- **data/ :** Folder to store input data files.
   - **reviews_segment.pkl**: Dataset of Amazon Reviews <span style="color:red; font-size:15px; font-weight:bold;">Not included, but required.</span>
   - **Topics.txt**: Topics file provided in project assignment. <span style="color:red; font-size:15px; font-weight:bold;">Not included, but required.</span>
- **nltk_data/**: Folder to download NLTK libraries. <span style="color:red; font-size:15px; font-weight:bold;">Not included, but required.</span>
- **opinion-lexicon-English/**: Folder to store opinion lexicon. <span style="color:red; font-size:15px; font-weight:bold;">Not included, but required.</span>
- **index/**: Folder containing pickle files.
- **output/**: Folder to store generated output files.


### Building Search Engine 
Run these commands to preprocess the data, build the inverted index, and set up search engine:
```bash
python preprocess.py
python build_index.py
python search_engine.py
```
**Note:**
- `preprocess.py` and `build_index.py` will take about **5-10 minutes** to complete.
- `search_engine.py` will take the longest because it has to build the files for Rating Search, Sentiment Analysis, and Topic Modeling. The files for Topic Modeling take the longest to build because has to train a Laten Dirichlet Allocation Model. Expect **10-20 minutes**.

### Running the Search
You can run the Boolean search using the provided `boolean_search.py` script. Here's an example:

```bash
python boolean_search_help.py --aspect1 "audio" --aspect2 "quality" --opinion "poor" --method "method1"
```

### Available Query Methods
1. **Method 1**: Boolean Search
2. **Method 2**: Rating Search
3. **Method 3**: Sentiment Analysis
4. **Method 4**: Topic Modeling

### Running Project Queries

Use the following commands to generate output files for the project:
```
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method1
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method2
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method3
python boolean_search_help.py --aspect1 audio --aspect2 quality --opinion poor --method method4
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method1
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method2
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method3
python boolean_search_help.py --aspect1 wifi --aspect2 signal --opinion strong --method method4
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method1
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method2
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method3
python boolean_search_help.py --aspect1 gps --aspect2 map --opinion useful --method method4
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method1
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method2
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method3
python boolean_search_help.py --aspect1 image --aspect2 quality --opinion sharp --method method4
```

## Contact

For any questions or issues, feel free to reach out:

- **Roman Guerra**
- **Email**: [rguerra6@cougarnet.uh.edu](mailto:rguerra6@cougarnet.uh.edu)
- **Teams**: Available via Teams