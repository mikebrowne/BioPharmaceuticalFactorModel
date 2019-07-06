'''

keyword_features_low_level_functions.py

'''

import pandas as pd
from itertools import combinations
from src.features.text_cleaning import clean_text, remove_company_name


# 1. Text Cleaning
def preprocess_text_data(articles_full, watchlist, column_name="title"):
    unique_companies = articles_full.ticker.unique()

    # General text cleaning
    cleaned_text = clean_text(articles_full, column_name)[column_name]

    # Remove company name from title
    company_names = watchlist.loc[watchlist.Ticker.isin(unique_companies)].index.tolist()
    return cleaned_text.apply(remove_company_name, args=(company_names,))


# 2. Feature Extraction
def extract_words(text_series):
    full_text_corpus = merge_text(text_series)

    unique_words = filter_unique_words(full_text_corpus)

    return get_token_frequency_dict(unique_words, text_series)


def merge_text(texts):
    return " ".join(texts.values)


def filter_unique_words(corpus):
    words = set(corpus.split(" "))
    words.remove("")
    return words


def get_token_frequency_dict(token_set, texts):
    return {token: get_token_frequency(token, texts) for token in token_set}


def get_token_frequency(token, texts):
    return sum([1 if token in text else 0 for text in texts.values]) / texts.shape[0]


def reduce_words_by_freq(word_frequency, cut_off=0.01):
    return [word for word in word_frequency.keys() if word_frequency[word] > cut_off]


def remove_short_strings(word_list, min_length=4):
    return [word for word in word_list if len(word) > min_length]


def extract_feature_vectors(text_series, features):
    return pd.DataFrame({ind: extract_features_boolean(text, features) for ind, text in text_series.iteritems()},
                        index=features).T


def extract_features_boolean(text, features):
    return [True if feature in text.split(" ") else False for feature in features]


# 3. Feature Engineering
def engineer_features(words, max_num_grams_per_token=2):
    tokens = []

    if max_num_grams_per_token is not None:
        for n in range(2, max_num_grams_per_token + 1):
            tokens += get_ngrams(words, n)

    return [tok for tok in tokens if type(tok) != str]


def get_ngrams(words, n):
    return combinations(words, n)


def extract_new_feature_vectors(new_feats, x_pre_eng):
    return pd.DataFrame({ind: is_feature_in_text(row, new_feats) for ind, row in x_pre_eng.iterrows()},
                        index=new_feats).T


def is_feature_in_text(row, features):
    return [extract_tuple_boolean(row, feature) for feature in features]


def extract_tuple_boolean(row, feature):
    return all([row[token] for token in feature])


# 4. Feature Reduction
def reduce_features_by_frequency(x, cut_off_value):

    features_filtered = x.sum().loc[x.sum() > cut_off_value].index

    return x[features_filtered]


def combine_features(list_of_features):
    return pd.concat(list_of_features, axis=1)
