'''

build_keyword_features.py

A pipeline for taking the raw article data and converting it into usable feature data for data
processing. See "04-KeywordAsAFactor_FeaturePipeline" for more information.

'''

from keyword_features_low_level_functions import *

# -------------------
# Full Pipeline
# -------------------

class FeatureCreationPipeline:
    def __init__(self):
        self.cut_off = 0
        self.feat_extractor = FeatureExtraction()
        self.feat_engineer = FeatureEngineering()
        self.feature_reducer = FeatureReduction()

        self.x = None

    def fit_transform(self, articles, watchlist, cut_off=None):
        if cut_off is not None:
            self.cut_off = cut_off

        titles_cleaned = preprocess_text_data(articles, watchlist, "title")
        x_pre_feat_eng = self.feat_extractor.fit_transform(titles_cleaned, cut_off)
        x_post_feat_red = self.feat_engineer.fit_transform(x_pre_feat_eng)

        self.x = self.feature_reducer.fit_transform([x_pre_feat_eng, x_post_feat_red], cut_off)


# -------------------
# Pipeline Objects
# -------------------

class FeatureExtraction:
    def __init__(self, min_str_length=4):
        self.cut_off = 0
        self.min_str_length = min_str_length
        self.word_list = []

    def fit(self, X, cut_off=0.01):
        self.cut_off = cut_off
        word_freq = extract_words(X)
        words_reduced_by_freq = reduce_words_by_freq(word_freq, self.cut_off)
        self.word_list += remove_short_strings(words_reduced_by_freq, self.min_str_length)

    def transform(self, X):
        return extract_feature_vectors(X, self.word_list)

    def fit_transform(self, X, cut_off=0.01, ):
        self.fit(X, cut_off)
        return self.transform(X)


class FeatureEngineering:
    def __init__(self, max_n_grams=2):
        self.max_n_grams = max_n_grams
        self.new_feature_list = []

    def fit(self, x):
        self.new_feature_list = engineer_features(x.columns.tolist(), max_n)

    def transform(self, x):
        return extract_new_feature_vectors(self.new_feature_list, x)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class FeatureReduction:
    def __init__(self):
        self.cut_off = 0
        self.cut_off_value = 0

    def fit(self, num_samples, cut_off=0.01):
        self.cut_off_value = self.cut_off * num_samples
        self.cut_off = cut_off

    def transform_individual(self, feature_set):
        return (reduce_features_by_frequency(feature_set, self.cut_off_value))

    def transform_batch(self, list_feature_sets):
        return combine_features([self.transform_individual(feature_set) for feature_set in list_feature_sets])

    def fit_transform(self, x, cut_off=0.01):
        if type(x) == list:
            num_samples = sum([list_item.shape[1] for list_item in x])
            self.fit(num_samples, cut_off)
            return self.transform_batch(x)

        else:
            num_sample = x.shape[1]
            self.fit(x, cut_off)
            return self.transform_individual(x)

# -------------------
# Low Level Functions
# -------------------

# 1. CLEAN DATA
def preprocess_text_data(articles_full, watchlist, column_name="title"):
    unique_companies = articles_full.ticker.unique()

    # General text cleaning
    cleaned_text = clean_text(articles_full, column_name)[column_name]

    # Remove company name from title
    company_names = watchlist.loc[watchlist.Ticker.isin(unique_companies)].index.tolist()
    return cleaned_text.apply(remove_company_name, args=(company_names,))


# 2. FEATURE EXTRACTION



# 3. FEATURE ENGINEERING



# 4. FEATURE REDUCTION
