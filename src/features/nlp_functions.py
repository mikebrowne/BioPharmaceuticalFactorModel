'''

nlp_functions.py

A collection of functions and classes to assist in NLP exploration and modelling.

    Functions Include:
        * remove_non_english_article

'''

from langdetect import detect


def remove_non_english_articles(df):
    '''Returns the df with all non-English articles removed.'''
    def detect_subsample(article):
        try:
            sub_article = article[50:250]
            return detect(sub_article)
        except Exception as e:
            return None

    return df.loc[df.article.apply(detect_subsample) == "en"]