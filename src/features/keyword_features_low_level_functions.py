'''

keyword_features_low_level_functions.py

'''

# 1. Feature Extraction
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
    return pd.DataFrame({ind: extract_features_boolean(text, features) for ind, text in text_series.iteritems()}, index=features).T


def extract_features_boolean(text, features):
    return [True if feature in text.split(" ") else False for feature in features]


# 2. Feature Engineering
def engineer_features(words, max_num_grams_per_token=2):
    tokens = []

    if max_num_grams_per_token is not None:
        for n in range(2, max_num_grams_per_token + 1):
            tokens += get_ngrams(words, n)

    return [tok for tok in tokens if type(tok) != str]


def get_ngrams(words, n):
    return combinations(words, n)


def extract_new_feature_vectors(new_feats, x_pre_eng):
    return pd.DataFrame({ind: extract_features_boolean(row, new_feats) for ind, row in x_pre_eng.iterrows()}, index=new_feats).T


def extract_features_boolean(row, features):
    return [extract_tuple_boolean(row, feature) for feature in features]


def extract_tuple_boolean(row, feature):
    return all([row[token] for token in feature])