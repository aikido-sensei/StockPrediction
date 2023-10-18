"""
Requirements:
- Given nasdaq_id fetch stock metrics on performance and company related info
    -From local database
    -From api
"""
import re
from re import Pattern
import pandas as pd
from pandas import Series
from functools import lru_cache
import json
import os.path
import requests
from datetime import date
from configparser import ConfigParser
from DataFetching import utils


class StockInfoFetcher:

    def __init__(self):
        self.resource_path: str = os.path.dirname(__file__)
        nasdaq_listing_path: str = os.path.join(
            self.resource_path,
            'nasdaq-listed_csv.csv'
        )
        self.stock_info_csv: pd.DataFrame = pd.read_csv(nasdaq_listing_path)
        self.symbol_info_csv: pd.DataFrame = pd.read_csv(
            nasdaq_listing_path,
            index_col="Symbol"
        )
        config_file_path: str = os.path.join(
            self.resource_path,
            'config.ini'
        )
        self.config: ConfigParser = ConfigParser()
        self.config.read(config_file_path)

    @lru_cache(maxsize=100)
    def get_commodity_fixed_info(self, nasdaq_id: str) -> dict:
        # Input validation.
        if not self.validate_commodity_id(nasdaq_id):
            raise AttributeError(f"Invalid NASDAQ ID - {nasdaq_id}")

        row: Series = self.symbol_info_csv.loc[nasdaq_id]
        info_on_commodity: dict = {
            "symbol": nasdaq_id,
            "company_name": row[0],
            "security_name": row[1],
            "market_category": row[2],
            "test_issue": row[3],
            "financial_status": row[4],
            "round_lot_size": row[5]
        }
        return info_on_commodity

    @lru_cache(maxsize=100)
    def validate_commodity_id(self, nasdaq_id: str) -> bool:
        if type(nasdaq_id) != str:
            return False
        nasdaq_id: str = nasdaq_id.upper()
        return nasdaq_id in self.stock_info_csv["Symbol"].values

    def get_all_time_daily_variable_commodity_info(self, nasdaq_id: str) -> dict:
        # Input validation.
        if not self.validate_commodity_id(nasdaq_id):
            raise AttributeError(f"Invalid NASDAQ ID - {nasdaq_id}")

        today: date = date.today()
        current_date: str = today.strftime("%Y-%m-%d")
        file_path: str = os.path.join(self.resource_path, f'{nasdaq_id}-{current_date}.json')
        if not os.path.exists(file_path):
            utils.delete_matching_files(self.resource_path, r'' + re.escape(nasdaq_id) + r'-\d{4}-\d{2}-\d{2}')
            key: str = self.config["alpha_vantage"]["api_key"]
            url: str = self.config["alpha_vantage"]["url"].format(nasdaq_id=nasdaq_id, key=key)
            response: requests.Response = requests.get(url)
            if not response.ok:
                raise AssertionError(f"API response code is {response.status_code} and raises the error : "
                                     f"{response.reason}.The API call failed")
            data: dict = response.json()
            with open(file_path, 'w', encoding='utf-8') as file_handle:
                json.dump(data, file_handle, ensure_ascii=False, indent=4)
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            data = json.load(file_handle)
        return data

    def get_daily_variable_commodity_info(self, nasdaq_id: str, day_needed: str) -> dict:
        # Input validation.
        if not self.validate_commodity_id(nasdaq_id):
            raise AttributeError(f"Invalid NASDAQ ID - {nasdaq_id}")
        day_needed_pattern: Pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        if not day_needed_pattern.match(day_needed):
            raise AttributeError(f"Invalid day_needed - {day_needed}")

        data = self.get_all_time_daily_variable_commodity_info(nasdaq_id)
        if day_needed in data["Time Series (Daily)"].keys():
            daily_info = data["Time Series (Daily)"][day_needed]
            return daily_info
        else:
            raise AttributeError(
                f"The specified date '{day_needed}' is invalid. The day might not be closed or too far back."
            )

    def get_current_variable_commodity_info(self, nasdaq_id: str) -> dict:
        key: str = self.config["fmp"]["api_key"]
        url: str = self.config["fmp"]["url"].format(nasdaq_id=nasdaq_id, key=key)
        response: requests.Response = requests.get(url)
        if not response.ok:
            raise AssertionError(f"API response code is {response.status_code} and raises the error : "
                                 f"{response.reason}.The API call failed")
        data: dict = response.json()[0]
        return data

    def reset(self) -> None:
        file_pattern: str = r'[a-zA-Z]{4}' + r'-\d{4}-\d{2}-\d{2}'
        utils.delete_matching_files(self.resource_path, file_pattern)


# Create an instance of StockInfoFetcher
x = StockInfoFetcher()

# Gets the complete lise of daily information of the stock AAPL and creates a json file
x.get_all_time_daily_variable_commodity_info('GOOG')

# Resets all the data of the fetcher and the files it created
#x.reset()

