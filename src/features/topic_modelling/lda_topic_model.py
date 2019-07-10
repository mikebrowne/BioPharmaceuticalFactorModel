# Imports

# Standard Libraries
from itertools import combinations

# Pickle
import pickle

# Numerical Libraries
import numpy as np
from scipy.stats import skew, kurtosis
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
import operator

# NLP Libraries
from nltk import word_tokenize, pos_tag
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
import pyLDAvis.gensim

# Local Package Libraries
import sys
sys.path.append("../..")

from src.data.make_dataset import *
from src.features.general_helper_functions import *
from src.features.text_cleaning import *
from src.features.topic_modelling.topic_model_helper_functions import *


class LDATopicModel:
    # adjust params to accept dictionary:
    # https://stackoverflow.com/questions/4989850/dictionary-input-of-function-in-python
    def __init__(self):
        # settings
        self.min_topics = 3
        self.max_topics = 5
        self.score = davies_bouldin_score

        # objects
        self.models = TopicModelHolder(self.score, "<")
        self.data = {}

    # PUBLIC METHODS
    def fit(self, articles, watchlist):
        self._get_metadata(articles, watchlist)
        self._clean_text()
        self._nlp_feature_engineering()
        self._build_models()

    def classify(self):
        pass

    # PRIVATE METHODS
    def _get_metadata(self, articles, watchlist):
        self.articles = articles
        self.watchlist = watchlist
        self.num_articles = self.articles.shape[0]
        self.data["initial"] = self.articles.title

    def _clean_text(self):
        # Settings and attributes
        list_pos_to_keep = [
            "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
            "JJ", "JJR", "JJS", "RB", "RBR", "RBS"
        ]

        list_of_company_names = self.watchlist.index.tolist()

        text = self.articles.copy()
        # 1. Standard Text Cleaning
        text = clean_text(text, "title").title
        # 2. Filter Part of Speech
        list_pos_to_keep = text_filter(text, list_pos_to_keep)
        text = text.apply(filter_pos_from_text, args=(list_pos_to_keep,))
        # 3. Lemmatize
        text = text.apply(lemmatize_text)
        # 4. Remove company names
        text = text.apply(remove_company_name, args=(list_of_company_names,))

        self.data["cleaned_data"] = text

    def _nlp_feature_engineering(self):
        tokenized_titles_cleaned = [text.split(" ") for text in self.data["cleaned_data"]]
        dictionary = corpora.Dictionary(tokenized_titles_cleaned)
        dictionary.filter_extremes(no_below=5, no_above=0.4)

        corpus = [dictionary.doc2bow(text) for text in tokenized_titles_cleaned]

        self.tfidf_corpus = TfidfModel(corpus)[corpus]

        self.data["features"] = gensim.matutils.corpus2dense(self.tfidf_corpus, self.num_articles)

    def _build_models(self):
        for model_n in range(self.min_topics, self.max_topics + 1):
            lda_n = LdaModel(self.tfidf_corpus, model_n, passes=1, iterations=50)

            best_labels = get_best_labels(lda_n, self.tfidf_corpus)

            self.models.update(self.data["features"], lda_n, model_n, best_labels)


def test():
    # raw data import
    DATA_PATH = os.path.join("..", "..", "..", "data")

    folder_path = os.path.join(DATA_PATH, "interim")
    file_name = "business_wire_scrape_results-clean_01.csv"
    articles = pd.read_csv(os.path.join(folder_path, file_name), index_col=0)

    folder_path = os.path.join(DATA_PATH, "raw")
    file_name = "watchlist_nasdaq_feb262019.csv"
    watchlist = pd.read_csv(os.path.join(folder_path, file_name), index_col=0)

    # Base data cleaning and formatting
    articles.reset_index(inplace=True)
    articles.time = pd.to_datetime(articles.time)

    lda_model = LDATopicModel()
    lda_model.fit(articles, watchlist)

    model_df = lda_model.models.model_df

    # Save
    # TODO: add a save method into the model class
    pickle_filename = "test_save"
    with open(os.path.join("..", "..", "models", pickle_filename), "wb") as outfile:
        pickle.dump(model_df, outfile)


if __name__ == "__main__":
    pd.set_option('display.max_colwidth', 1000)
    test()

    pickle_filename = os.path.join("..", "..", "models", "test_save")
    with open(pickle_filename, 'rb') as infile:
        model_df = pickle.load(infile)

    print(model_df)
