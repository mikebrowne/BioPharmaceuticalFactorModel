import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.features.topic_modelling.topic_model_helper_functions import sklearn_get_topic_terms


def highlight_repeat_words(lda_holder, num_topics=3, num_keywords=3):
    top_words_df = pd.DataFrame(sklearn_get_topic_terms(lda_holder, num_topics, num_keywords))

    words_in_multiple_topics = top_words_df.stack().value_counts().loc[top_words_df.stack().value_counts() > 1].index.tolist()

    styled_top_words_df = top_words_df.style.apply(highlight_word, args=(words_in_multiple_topics,))

    return styled_top_words_df


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def highlight_word(s, word_list):
    '''
    highlight the maximum in a Series yellow.
    '''
    colors = sns.light_palette((210, 90, 60), input="husl", n_colors=len(word_list)).as_hex()
    df = s.copy()
    df.loc[~df.isin(word_list)] = ""
    for i, word in enumerate(word_list):
        df.loc[df == word] = "background-color: {}".format(colors[i])

    return df


def topic_result_histogram(articles, watchlist, lda_holder, num_topics, ax):
    ax.set_title("Number of Articles for Each Topic")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Articles")
    topic_results = pd.Series(lda_holder.get_topic(articles, watchlist, num_topics))
    val_counts = topic_results.value_counts()
    return sns.barplot(x=val_counts.index, y=val_counts, palette=sns.light_palette((210, 90, 60), input="husl", reverse=True), ax=ax)