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
from gensim.test.utils import datapath

# Local Package Libraries
from src.data.make_dataset import *
from src.features.text_cleaning import *
from src.features.topic_modelling.topic_model_helper_functions import *
from src.utils.pickle_compression import save_pickle_compress, load_pickle_compress


class PreProcessing:
    '''Cleans, formats and transforms the data into TD-IDF vectors.'''
    def __init__(self):
        pass

    # Public Functions
    def fit_from_data(self, article, watchlist):
        '''
        The initial fitting of the pre-processing. Creates the Dictionary, Corpus, and TF-IDF matrix.
        :param article: (pd.DataFrame) of articles, specifically with the column "title"
        :param watchlist: (pd.DataFrame) specifically with columns "ticker" and "company_name"
        :return: GenSim Corpus, Dictionary, TFIDF
        '''
        articles_cleaned = self._clean_text(article, watchlist)
        dict, tfidf_model, tfidf_corpus = self._nlp_feature_engineering(articles_cleaned)
        self.dict_ = dict
        self.tfidf_ = tfidf_model
        return dict, tfidf_model, tfidf_corpus

    def fit_from_pickle(self, pickle_list):
        '''
        Re-fits the Pre-Processing for transforming data to work in the fitted LDATopicModel.
        :param pickle_list: pickled GenSim objects
        :return: None
        '''
        self.dict_ = open_pickle(pickle_list[0])
        self.tfidf_ = open_pickle(pickle_list[1])

    def transform(self, articles, watchlist):
        '''transforms the article into a TF-IDF vector for input into the LDATopicModel'''
        if type(watchlist) == pd.DataFrame:
            watchlist = watchlist.index.tolist()
        # otherwise it will be a list of company names

        cleaned_articles = self._clean_text(articles, watchlist)
        tokenized_titles_cleaned = [text.split(" ") for text in cleaned_articles]
        new_corpus = [self.dict_.doc2bow(text) for text in tokenized_titles_cleaned]

        return self.tfidf_[new_corpus]

    # Private Functions
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

    @staticmethod
    def _nlp_feature_engineering(texts):
        tokenized_titles_cleaned = [text.split(" ") for text in texts]

        dictionary = corpora.Dictionary(tokenized_titles_cleaned)
        dictionary.filter_extremes(no_below=20, no_above=0.4)

        corpus = [dictionary.doc2bow(text) for text in tokenized_titles_cleaned]

        tfidf_model = TfidfModel(corpus)
        tfidf_corpus = tfidf_model[corpus]

        return dictionary, tfidf_model, tfidf_corpus


class LDATopicModel:
    '''LDA model that is fitted with TF-IDF vectors of a set of articles.'''
    def __init__(self, min_topics=3, max_topics=4, n_passes=1, n_iterations=50):
        # settings
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.n_passes = n_passes
        self.n_iterations = n_iterations
        self.score = silhouette_score

        # objects
        self.models = TopicModelHolder(self.score, ">")

    def fit(self, X, tfidf_corpus):
        '''
        Fits the LDA model
        :param X:
        '''

        X = gensim.matutils.corpus2dense(tfidf_corpus, len(X))

        for model_n in range(self.min_topics, self.max_topics + 1):
            lda_n = LdaModel(tfidf_corpus, model_n, passes=self.n_passes, iterations=self.n_iterations)

            best_labels = get_best_labels(lda_n, tfidf_corpus)

            self.models.update(X, lda_n, model_n, best_labels)

        self.model = self.models.get_best_model()
        del self.models

    def get_topic(self, X):
        '''
        Predicts the topic using a fitted LDA Model
        :param X: (GenSim TFIDF)
        :param model_n: the number of topics
        :return: topic number
        '''
        def get_best_probability(list_tuples):
            best_prob = 0
            best_topic = None
            for item in list_tuples:
                if item[1] > best_prob:
                    best_prob = item[1]
                    best_topic = item[0]

            return best_topic

        topic_output_with_probs = [self.model.get_document_topics(x_i) for x_i in X]
        topic_results = [get_best_probability(tuple_) for tuple_ in topic_output_with_probs]

        return topic_results


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
    # First section will act as though it is in file_1, the second section as though in file_2

    # -----------
    # file_1.py
    # -----------
    articles, watchlist = load()

    print("Fitting Pre Processing...")
    p = PreProcessing()
    dict, tfidf_model, tfidf_corpus = p.fit_from_data(articles, watchlist)

    save_pickle(dict, "lda_model_dict")
    save_pickle(tfidf_model, "lda_model_tfidf_model")

    print("File size of {}: ".format("lda_model_dict"), os.path.getsize(get_path("lda_model_dict")))
    print("File size of {}: ".format("lda_model_tfidf_model"), os.path.getsize(get_path("lda_model_tfidf_model")))

    X = p.transform(articles, watchlist)

    print("Training LDA Model...")
    model = LDATopicModel(max_topics=30, n_passes=30, n_iterations=50)
    model.fit(X, tfidf_corpus)

    save_pickle(model, "lda_model")
    print("File size of {}: ".format("lda_model"), os.path.getsize(get_path("lda_model")))

    # path = get_path("lda_model_save")
    # temp_file = datapath(path)
    # model.models.save(fname=temp_file)
    # print("File size of {}: ".format("lda_model_save"), os.path.getsize(get_path("lda_model_save")))

    print("DONE file_1...\n\n")

    # ------------
    # file_2.py
    # ------------

    print("Re-fitting Pre Processing...")
    new_p = PreProcessing()
    new_p.fit_from_pickle(["lda_model_dict", "lda_model_tfidf_model"])

    model = open_pickle("lda_model")

    article = articles.sample(1)

    print("Applying Transform to Article, Then Predict the Topic...")
    x_test = new_p.transform(article, watchlist)

    topic = model.get_topic(X=x_test)

    print("The topic is: ",topic)
