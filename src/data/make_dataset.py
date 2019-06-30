'''

make_dataset.py

Functions for importing data and sending the data through the feature engineering
pipeline.

    Functions Include:
        * get_raw_data

'''
import os
import sys
import pandas as pd

DATA_PATH = "../../data"


def get_raw_data():
    '''
    Returns the raw data:
        * business wire scraper results
        * watchlist of pharmaceutical related companies (since Feb 26, 2019
        * stock prices for the companies on the watchlist
    '''
    file_name_1 = "business_wire_scrape_results.csv"
    file_name_2 = "watchlist_nasdaq_feb262019.csv"
    file_name_3 = "stock_prices_asof_2019-06-21.csv"

    file_1 = pd.read_csv(os.path.join(DATA_PATH, "raw", file_name_1), index_col=0)
    file_2 = pd.read_csv(os.path.join(DATA_PATH, "raw", file_name_2), index_col=0)
    file_3 = pd.read_csv(os.path.join(DATA_PATH, "raw", file_name_3), index_col=0)

    return file_1, file_2, file_3


def main():
    pass


if __name__ == '__main__':
    main()
