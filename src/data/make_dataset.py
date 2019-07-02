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

from src.features.text_cleaning import remove_non_english_articles

DATA_PATH = os.path.join("..", "..", "data")


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


def clean_and_open_business_wire_data_01(df, to_save=False):
    '''Cleans Business Wire data for First Data Exploration'''

    file_name = "business_wire_scrape_results-clean_01.csv"
    folder_path = os.path.join(DATA_PATH, "interim")

    try:
        temp_df = pd.read_csv(os.path.join(folder_path, file_name), index_col=0)
        if df is None:
            return temp_df
        if temp_df.shape == df.shape:
            return temp_df
    except Exception as e:
        print(str(e))

    clinical_trial_df = df.copy()

    # 1: Remove NaN
    clinical_trial_df.dropna(inplace=True)

    # 2: Remove non-English articles
    clinical_trial_df = remove_non_english_articles(clinical_trial_df)

    # 3: Set all strings to lower case in "title" and "article" columns
    clinical_trial_df.article = clinical_trial_df.article.apply(str.lower)
    clinical_trial_df.title = clinical_trial_df.title.apply(str.lower)

    # 4: Drop "link" column
    if "link" in clinical_trial_df.columns:
        clinical_trial_df.drop("link", inplace=True, axis=1)

    # 5: Ensure date is a datetime object
    clinical_trial_df.time = pd.to_datetime(clinical_trial_df.time)

    if to_save:
        clinical_trial_df.to_csv(os.path.join(folder_path, file_name))

    return clinical_trial_df


def clean_and_format_watchlist(watchlist_df, company_tickers):

    # 1: Keep only the companies from the list
    watchlist_df = watchlist_df.loc[watchlist_df.Ticker.isin(company_tickers)]

    # 2: Change column names to all lowercase
    watchlist_df.columns = ["ticker", "marketcap", "sector", "exchange"]

    return watchlist_df


def clean_and_format_prices(prices_df, company_tickers):

    # 1: Reduce the new copy of prices to only the companies under our scope
    prices_df = prices_df[company_tickers]

    # 2: Sort by date
    prices_df.sort_index(inplace=True)

    # 3: Ensure index is datetime object
    prices_df.index = pd.to_datetime(prices_df.index)

    return prices_df


def main():
    bus_wire_raw, _, _ = get_raw_data()
    temp_df = clean_and_open_business_wire_data_01(bus_wire_raw, to_save=True)
    print(temp_df)


if __name__ == '__main__':
    sys.path.append("../..")
    main()
