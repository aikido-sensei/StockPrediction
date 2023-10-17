from abc import abstractmethod


class CommodityInfoFetcher:
    """
    Used as a template for the different InfoFetchers. It is meant to make sure that the InfoFetcher have the following
    functions.
    """

    @abstractmethod
    def get_commodity_fixed_info(self, commodity_id: str) -> dict:  # pragma: no cover
        """
        Get the info on the required commodity
        :param commodity_id: Commodity id is a string. It is the name by which the commodity is referenced on the
        market.
        :return: It returns a dictionary containing the fixed information on the commodity.
        """
        pass

    @abstractmethod
    def validate_commodity_id(self, commodity_id: str) -> bool:  # pragma: no cover
        """
        This function checks whether the commodity id given is valid in a certain market or not.
        :param commodity_id: Commodity id is a string. It is the name by which the commodity is referenced on the
        market.
        :return: True if the commodity can be found on the specific market, else False
        """
        pass

    @abstractmethod
    def get_daily_variable_commodity_info(self, commodity_id: str, day_needed: str) -> dict:  # pragma: no cover
        """
        Gets the variable commodity info on specified commodity at on a specified day.
        :param commodity_id: Commodity id is a string. It is the name by which the commodity is referenced on the
        market.
        :param day_needed: A string of the day for which the commodity's info are needed. It is set in nanoseconds.
        Format is year-month-day
        :return: It returns a dictionary containing the variable information on the commodity for the specified day.
        """
        pass

    @abstractmethod
    def get_all_time_daily_variable_commodity_info(self, commodity_id: str) -> dict:  # pragma: no cover
        """
        Gets the info on the commodity for every day that's available and saves the data in a json file if it needs
         to be updated and also returns the most recent data.
        :param commodity_id: Commodity id is a string. It is the name by which the commodity is referenced on the
        market.
        :return: It returns a dictionary containing all the daily data available on the specified commodity id.
        """
        pass

    @abstractmethod
    def get_current_variable_commodity_info(self, commodity_id: str) -> dict:  # pragma: no cover
        """
        Gets the current volume and the current price of  the specified commodity_id.
        :param commodity_id: Commodity id is a string. It is the name by which the commodity is referenced on the
        market.
        :return: It returns a dict containing the current price, the commodity id and the current volume.
        """
        pass

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover
        """
        Clears all existing data on disk.
        """
        pass
