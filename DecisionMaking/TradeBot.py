from abc import abstractmethod


class TradeBot:
    """
    Used as a template for the different algorithms. It is meant to make sure that the algorithms for trade decisions
    have the following functions.
    """

    def __init__(self, parameters: dict) -> None:  # pragma: no cover
        """
        Takes the parameters needed as input and initializes them as a class dictionary and then tests their validity.
        :param parameters: A dictionary containing the different parameters passed to the model.
        """
        self.parameters: dict = parameters
        self.validate_parameters()

    @abstractmethod
    def train(self, data: dict) -> None:  # pragma: no cover
        """
        Trains the model.
        :param data: Data on which the model is trained.
        """
        pass

    @abstractmethod
    def test(self, data: dict) -> dict:  # pragma: no cover
        """
        Tests the model's accuracy.
        :param data: Data on which the model is tested.
        :return: A dict containing the different indicators of the accuracy of the model.
        """
        pass

    @abstractmethod
    def validate_parameters(self) -> None:  # pragma: no cover
        """
        Validates the parameters given in the model and raises an error if they are invalid.
        """
        pass

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover
        """
        Deletes all data for TradeBot class
        """
        pass
