'''

text_cleaning.py

A collection of functions and classes to assist in NLP exploration and modelling.

    Functions Include:
        * remove_non_english_article
        * remove_white_spaces
        * remove_non_alphanumeric
        * remove_numbers
        * remove_stop_words

'''

import re

from langdetect import detect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pandas import Series


def remove_non_english_articles(df):
    '''Returns the df with all non-English articles removed.'''
    def detect_subsample(article):
        try:
            sub_article = article[50:250]
            return detect(sub_article)
        except Exception as e:
            return None

    return df.loc[df.article.apply(detect_subsample) == "en"]


def remove_white_spaces(text):
    '''Removes all white spaces that are not a single space between words.'''
    replace = {
        ord('\f'): ' ',
        ord('\t'): ' ',
        ord('\n'): ' ',
        ord('\r'): None,
    }

    return text.translate(replace)


def remove_non_alphanumeric(text):
    # Not sure why I can't get RegEx to work here for the full article so will implement a work around and fix later
    return " ".join([re.sub(r'\W+', ' ', word_token) for word_token in text.split(" ")])


def remove_numbers(text):
    return re.sub('[0-9]+', '', text)


def remove_stop_words(text):
    stop_words = stopwords.words('english')
    return " ".join([word for word in text.split(" ") if word not in stop_words])


def remove_company_name(text, company_names):
    set_of_words = set()
    for name in company_names:
        for sub_name in name.split(" "):
            set_of_words.add(sub_name.lower())

    return " ".join([word for word in text.split(" ") if word not in set_of_words])


def remove_short_words(text, min_len=4):
    return " ".join([word for word in text.split(" ") if len(word) >= min_len])


def lemmatize_text(text):
    lemmetizer = WordNetLemmatizer()
    tokens = text.split(" ")
    tokens = [lemmetizer.lemmatize(word) for word in tokens]
    tokens = [lemmetizer.lemmatize(word, pos="v") for word in tokens]
    return " ".join(tokens)


def clean_text(df, column_name):
    df[column_name] = (
        df[column_name].apply(remove_white_spaces)
            .apply(remove_non_alphanumeric)
            .apply(remove_numbers)
            .apply(remove_stop_words)
    )
    return df


if __name__=="__main__":
    print(lemmatize_text("accounts"))
    print(lemmatize_text("announces"))