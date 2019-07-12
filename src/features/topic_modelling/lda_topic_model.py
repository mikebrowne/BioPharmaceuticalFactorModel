# Imports

# Standard Libraries
from itertools import combinations

# Pickle
import pickle

# Numerical Libraries
from sklearn.metrics import *
import pandas as pd

# NLP Libraries
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel

# Local Package Libraries
# import sys
# sys.path.append("../..")

from src.data.make_dataset import *
from src.features.text_cleaning import *
from src.features.topic_modelling.topic_model_helper_functions import *


class LDATopicModel:
    # adjust params to accept dictionary:
    # https://stackoverflow.com/questions/4989850/dictionary-input-of-function-in-python
    def __init__(self):
        # settings
        self.min_topics = 3
        self.max_topics = 30
        self.score = davies_bouldin_score

        # objects
        self.models = TopicModelHolder(self.score, "<")
        self.data = {}

    # PUBLIC METHODS
    def fit(self, articles, watchlist):
        self._get_metadata(articles, watchlist)
        self.data["cleaned_data"] = self._clean_text(self.articles, self.watchlist)
        self._nlp_feature_engineering()
        self._build_models()

    def transform(self, topic_number, articles, watchlist):
        def get_best_probability(list_tuples):
            best_prob = 0
            best_topic = None
            for item in list_tuples:
                if item[1] > best_prob:
                    best_prob = item[1]
                    best_topic = item[0]

            return best_topic

        if type(watchlist) == pd.DataFrame:
            watchlist = watchlist.index.tolist()

        cleaned_articles = self._clean_text(articles, watchlist)
        tokenized_titles_cleaned = [text.split(" ") for text in cleaned_articles]
        new_corpus = [self.dictionary.doc2bow(text) for text in tokenized_titles_cleaned]
        new_tfidf_corpus = self.tfidf_model[new_corpus]

        model = self.models.model_df.loc[topic_number].model

        topic_output_with_probs = [model.get_document_topics(x_i) for x_i in new_tfidf_corpus]
        topic_results = [get_best_probability(tuple_) for tuple_ in topic_output_with_probs]

        return topic_results

    def save(self, filename):
        with open(os.path.join("..", "..", "models", filename), "wb") as outfile:
            pickle.dump(self.models.model_df, outfile)

    # PRIVATE METHODS
    def _get_metadata(self, articles, watchlist):
        self.articles = articles
        self.watchlist = watchlist.index.tolist()
        self.num_articles = self.articles.shape[0]
        self.data["initial"] = self.articles.title

    def _clean_text(self, text, watchlist):
        # Settings and attributes
        list_pos_to_keep = [
            "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
            "JJ", "JJR", "JJS", "RB", "RBR", "RBS"
        ]

        list_of_company_names = watchlist

        # 1. Standard Text Cleaning
        text = clean_text(text, "title").title
        # 2. Filter Part of Speech
        list_pos_to_keep = text_filter(text, list_pos_to_keep)
        text = text.apply(filter_pos_from_text, args=(list_pos_to_keep,))
        # 3. Lemmatize
        text = text.apply(lemmatize_text)
        # 4. Remove company names
        text = text.apply(remove_company_name, args=(list_of_company_names,))

        return text

    def _nlp_feature_engineering(self):
        tokenized_titles_cleaned = [text.split(" ") for text in self.data["cleaned_data"]]
        dictionary = corpora.Dictionary(tokenized_titles_cleaned)
        dictionary.filter_extremes(no_below=5, no_above=0.4)

        self.dictionary = dictionary

        self.corpus = [dictionary.doc2bow(text) for text in tokenized_titles_cleaned]

        self.tfidf_model = TfidfModel(self.corpus)

        self.tfidf_corpus = self.tfidf_model[self.corpus]

        self.data["features"] = gensim.matutils.corpus2dense(self.tfidf_corpus, self.num_articles)

    def _build_models(self):
        for model_n in range(self.min_topics, self.max_topics + 1):
            lda_n = LdaModel(self.tfidf_corpus, model_n, passes=30, iterations=500)

            best_labels = get_best_labels(lda_n, self.tfidf_corpus)

            self.models.update(self.data["features"], lda_n, model_n, best_labels)


def test(to_save=False):
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

    print(lda_model.models.model_df)

    print("TEST:")

    topic_number = 3
    topics = lda_model.transform(topic_number, articles.sample(1), watchlist)
    print(topics)

    # Save
    if to_save:
        pickle_fname = os.path.join("..", "..", "models", "lda_model")
        with open(pickle_fname, 'wb') as outfile:
            pickle.dump(lda_model.models, outfile)


if __name__ == "__main__":
    pd.set_option('display.max_colwidth', 1000)

    to_save = True
    test(to_save)

    if to_save:
        pickle_filename = os.path.join("..", "..", "models", "lda_model")
        with open(pickle_filename, 'rb') as infile:
            model_df = pickle.load(infile)

        print(model_df)
