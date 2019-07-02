'''

general_helper_functions.py

Contains functions that will assist in creating features but do not necessarily warrant their own moduel

    Classes Contained:
        * GetPrices

'''
import pandas as pd
import numpy as np


class GetPrices:
    '''
    Class to get the prices surrounding an article release

    init
    ----
        * article_data_frame
        * price_data_frame
        * n_window
        * inplace

    methods
    -------
        * get_price
        * add_prices_to_frame

    '''
    def __init__(self, article_data_frame, price_data_frame, n_window=10, inplace=False):
        self.price_data_frame = price_data_frame
        self.article_data_frame = article_data_frame
        self.n_window = n_window
        self.inplace = inplace

        return

    def get_price(self, x):
        try:
            ticker, time = x.ticker, x.time
            # Since there are days that stocks do not trade (holidays and weekends) need to get the index of the
            # event date, then get the following n_window days using iloc
            index_number = self.price_data_frame.index.tolist().index(time)

            vals = self.price_data_frame[ticker].iloc[index_number: index_number + self.n_window].values

            return vals
        except Exception as e:
            # Will do better exception handling later. Currently just trying to get everything to work the right way.
            return [None] * self.n_window

    def add_prices_to_frame(self):
        '''Builds a dataframe containing the prices with the index from the articles'''
        new_data = self.article_data_frame.apply(lambda x: pd.Series(self.get_price(x)), axis=1)

        new_columns = ["P_{}".format(i) for i in range(self.n_window)]

        if self.inplace:
            self.article_data_frame[new_columns] = new_data

            # return self.article_data_frame
        else:
            new_data.columns = new_columns
            return new_data


def compute_return_window(article_df, prices_df, n_window=30):
    # Define percent return function here. Will move outside if it is needed for other functions
    def percent_return(value_matrix, i, j):
        return (value_matrix[j][i] / value_matrix[j][0]) - 1

    # Get the stock prices for "n" days following each event
    price_window = GetPrices(
        article_df,
        prices_df,
        n_window=n_window
    ).add_prices_to_frame()

    return_values = np.array([
        np.array([
            percent_return(price_window.values, i, j) for i in range(1, price_window.shape[1])
        ]) for j in range(price_window.shape[0])
    ])

    cols = ["R_{}".format(i) for i in range(return_values.shape[1])]

    return pd.DataFrame(return_values, index=price_window.index, columns=cols)