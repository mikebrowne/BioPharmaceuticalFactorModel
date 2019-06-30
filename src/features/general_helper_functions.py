'''

general_helper_functions.py

Contains functions that will assist in creating features but do not necessarily warrant their own moduel

    Classes Contained:
        * GetPrices

'''
from pandas import Series


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
        new_data = self.article_data_frame.apply(lambda x: Series(self.get_price(x)), axis=1)

        new_columns = ["P_{}".format(i) for i in range(self.n_window)]

        if self.inplace:
            self.article_data_frame[new_columns] = new_data

            # return self.article_data_frame
        else:
            new_data.columns = new_columns
            return new_data
