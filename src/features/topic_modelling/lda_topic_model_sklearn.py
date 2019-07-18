# Imports

# Standard Libraries
from itertools import combinations

# Pickle
import pickle
import sys

# Numerical Libraries
from sklearn.metrics import *
import pandas as pd

# NLP Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Local Package Libraries
from src.data.make_dataset import *
from src.features.text_cleaning import *
from src.features.topic_modelling.topic_model_helper_functions import *
from src.utils.pickle_compression import save_pickle_compress, load_pickle_compress


class LDATopicModel:

    def __init__(self, min_topics=3, max_topics=4, n_iterations=10):
        # settings
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.n_iterations = n_iterations
        self.score = calinski_harabasz_score

        # objects
        self.models = TopicModelHolder(self.score, ">")

    # PUBLIC FUNCTIONS
    # ----------------

    def fit(self, articles_, watchlist_):
        articles_cleaned = self._clean_text(articles_, watchlist_)
        self._fit_count_vec(articles_cleaned)
        tfidf_data = self._fit_tfidf(articles_cleaned)
        self._fit_models(tfidf_data)

    def get_topic(self, articles_, watchlist_, n_topic=3):
        return [np.argmax(x) for x in self.get_document_topic_probabilities(articles_, watchlist_, n_topic=3)]

    def get_document_topic_probabilities(self, articles_, watchlist_, n_topic=3):
        articles_cleaned = self._clean_text(articles_, watchlist_)
        tfidf_data = self.tfidf_model.transform(articles_cleaned)
        return self.models.model_df.loc[n_topic].model.transform(tfidf_data)

    # PRIVATE FUNCTIONS
    # -----------------

    @staticmethod
    def _clean_text(texts, watchlist):
        # Settings and attributes
        list_pos_to_keep = [
            "NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
            "JJ", "JJR", "JJS", "RB", "RBR", "RBS"
        ]

        list_of_company_names = watchlist

        # 1. Standard Text Cleaning
        text = clean_text(texts, "title").title
        # 2. Filter Part of Speech
        list_pos_to_keep = text_filter(text, list_pos_to_keep)
        text = text.apply(filter_pos_from_text, args=(list_pos_to_keep,))
        # 3. Lemmatize
        text = text.apply(lemmatize_text)
        # 4. Remove company names
        text = text.apply(remove_company_name, args=(list_of_company_names,))

        return text.values

    def _fit_count_vec(self, cleaned_articles_):
        self.count_vec = CountVectorizer(min_df=0.01, max_df=0.4)
        self.count_vec.fit(cleaned_articles_)

    def _fit_tfidf(self, cleaned_articles_):

        self.tfidf_model = TfidfVectorizer(**self.count_vec.get_params())

        return self.tfidf_model.fit_transform(cleaned_articles_)

    def _fit_models(self, tfidf_vectors_):
        for model_n in range(self.min_topics, self.max_topics + 1):
            lda_n = LDA(n_components=model_n,
                        max_iter=self.n_iterations)

            x_transformed = lda_n.fit_transform(tfidf_vectors_)

            best_labels = [np.argmax(x) for x in x_transformed]

            self.models.update(x_transformed, lda_n, model_n, best_labels)


# USEFUL FUNCTIONS
# ----------------

def load():
    # raw data import
    DATA_PATH = os.path.join("..", "..", "..", "data")

    folder_path = os.path.join(DATA_PATH, "interim")
    file_name = "business_wire_scrape_results-clean_01.csv"
    articles = pd.read_csv(os.path.join(folder_path, file_name), index_col=0)

    folder_path = os.path.join(DATA_PATH, "raw")
    file_name = "watchlist_nasdaq_feb262019.csv"
    watchlist = pd.read_csv(os.path.join(folder_path, file_name), index_col=0)
    return articles, watchlist


def get_path(object_file_name):
    return os.path.join("..", "..", "models", object_file_name)


def save_pickle(object, object_file_name):
    pickle_fname = get_path(object_file_name)
    with open(pickle_fname, 'wb') as outfile:
        pickle.dump(object, outfile)


def open_pickle(object_file_name):
    pickle_filename = get_path(object_file_name)
    with open(pickle_filename, 'rb') as infile:
        object = pickle.load(infile)

    return object


if __name__ == "__main__":

    articles, watchlist = load()

    lda_model = LDATopicModel(min_topics=3, max_topics=10)

    lda_model.fit(articles, watchlist)

    save_pickle(lda_model, "lda_model_sklearn")
    print("File size of {}: ".format("lda_model_sklearn"), os.path.getsize(get_path("lda_model_sklearn")))

    LDA_holder = open_pickle("lda_model_sklearn")

    dict_ = {"date": 50,
             "title": "ACADIA Pharmaceuticals Initiates Phase 3 CLARITY Program with Pimavanserin as Adjunctive Treatment for Major Depressive Disorder"}
    a = pd.DataFrame(dict_, index=[0])

    dict_ = {"date": 50, "title": "ACADIA Pharmaceuticals Reports First Quarter 2019 Financial Results"}
    b = pd.DataFrame(dict_, index=[0])

    dict_ = {"date": 50,
             "title": "ACADIA Pharmaceuticals Initiates Phase 3 CLARITY Program with Pimavanserin as Adjunctive Treatment for Major Depressive Disorder"}
    c = pd.DataFrame(dict_, index=[0])

    dict_ = {"date": 50,
             "title": "ACADIA Pharmaceuticals Initiates Phase 3 CLARITY Program with Pimavanserin as Adjunctive Treatment for Major Depressive Disorder"}
    d = pd.DataFrame(dict_, index=[0])

    for data in [a, b, c, d]:
        print(LDA_holder.get_topic(data, watchlist, 6))

    print(LDA_holder.get_topic(articles.sample(5), watchlist, 6))
