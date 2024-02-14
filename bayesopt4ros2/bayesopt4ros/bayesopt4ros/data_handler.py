from __future__ import annotations

import rclpy
import torch
import yaml

from torch import Tensor
from typing import Dict, Tuple, Union, List

# from botorch.utils.containers import TrainingData
# from botorch.utils.containers import DenseContainer
from botorch.utils.datasets import SupervisedDataset
from botorch.exceptions.errors import BotorchTensorDimensionError


class DataHandler(object):
    """Helper class that handles all data for BayesOpt.

    .. note:: This is mostly a convenience class to clean up the BO classes.
    """

    def __init__(
        self, x: Tensor = None, y: Tensor = None, maximize: bool = True, feature_names:list[str] = None , outcome_names:list[str] = None
    ) -> None:
        """The DataHandler class initializer.

        Parameters
        ----------
        x : torch.Tensor
            The training inputs.
        y : torch.Tensor
            The training targets.
        maximize : bool
            Specifies if 'best' refers to min or max.
        """
        # rclpy.logging.get_logger("dbg_dataHandler").info(f"dbg Data_handler: __init__ feat_names: {feature_names}, outcome_names:{outcome_names}")

        self.feature_names = feature_names
        self.outcome_names = outcome_names
        self.set_xy(x=x, y=y)
        self.maximize = maximize

    @classmethod
    def from_file(cls, file: Union[str, List[str]], feature_names, outcome_names) -> DataHandler:
        """Creates a DataHandler instance with input/target values from the
        specified file.

        Parameters
        ----------
        file : str or List[str]
            One or many evaluation files to load data from.

        Returns
        -------
        :class:`DataHandler`
            An instance of the DataHandler class. Returns an empty object if
            not file could be found.
        """
        files = [file] if isinstance(file, str) else file
        x, y = [], []
        # rclpy.logging.get_logger("dbg_dataHandler").info(f"dbg Data_handler: from_file: feat_names: {feature_names}, outcome_names:{outcome_names}")
        for file in files:
            try:
                with open(file, "r") as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                x.append(torch.tensor(data["train_inputs"],dtype=torch.double))
                y.append(torch.tensor(data["train_targets"],dtype=torch.double))
            except FileNotFoundError:
                rclpy.logwarn(f"The evaluations file '{file}' could not be found.")  #TODO: replace with rclpy.logging.get_logger().warnining('')

        if x and y:
            if (
                not len(set([xi.shape[1] for xi in x])) == 1
            ):  # check for correct dimension
                message = "Evaluation points seem to have different dimensions."
                raise BotorchTensorDimensionError(message)
            x = torch.cat(x)
            y = torch.cat(y)
            return cls(x=x, y=y,feature_names=feature_names, outcome_names=outcome_names)
        else:
            return cls(feature_names=feature_names,outcome_names=outcome_names)

    def get_xy(
        self, as_dict: dict = False
    ) -> Union[Dict, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the data as a tuple (default) or as a dictionary."""
        if as_dict:
            return {"train_inputs": self.data.X, "train_targets": self.data.Y}
        else:
            return (self.data.X, self.data.Y)

    def set_xy(self, x: Tensor = None, y: Union[float, Tensor] = None):
        """Overwrites the existing data."""
        if x is None or y is None:
            self.data = SupervisedDataset(X=torch.tensor([],dtype=torch.double), Y=torch.tensor([],dtype=torch.double),feature_names = [], outcome_names=[])
            #old: self.data = TrainingData(Xs=torch.tensor([]), Ys=torch.tensor([]))
        else:
            if not isinstance(y, Tensor):
                y = torch.tensor([[y]],dtype=torch.double)
            self._validate_data_args(x, y)
            #old: self.data = TrainingData(Xs=x, Ys=y)
            self.data = SupervisedDataset(X=x, Y=y,feature_names=self.feature_names, outcome_names=self.outcome_names)

    def add_xy(self, x: Tensor = None, y: Union[float, Tensor] = None):
        """Adds new data to the existing data."""
        if not isinstance(y, Tensor):
            y = torch.tensor([[y]],dtype=torch.double)
        x = torch.atleast_2d(x)
        self._validate_data_args(x, y)
        x = torch.cat((self.data.X, x)) if self.n_data else x
        y = torch.cat((self.data.Y, y)) if self.n_data else y
        self.set_xy(x=x, y=y)

    @property
    def n_data(self):
        """Number of data points."""
        return self.data.X.shape[0]

    @property
    def x_best(self):
        """Location of the best observed datum."""
        if self.maximize:
            return self.data.X[torch.argmax(self.data.Y)]
        else:
            return self.data.X[torch.argmin(self.data.Y)]

    @property
    def x_best_accumulate(self):
        """Locations of the best observed datum accumulated along first axis."""
        return self.data.X[self.idx_best_accumulate]  

    @property
    def y_best(self):
        """Function value of the best observed datum."""
        if self.maximize:
            return torch.max(self.data.Y)
        else:
            return torch.min(self.data.Y)

    @property
    def y_best_accumulate(self):
        """Function value of the best ovbserved datum accumulated along first axis."""
        return self.data.Y[self.idx_best_accumulate]

    @property
    def idx_best_accumulate(self):
        """Indices of the best observed data accumulated along first axis."""
        argminmax = torch.argmax if self.maximize else torch.argmin
        return [argminmax(self.data.Y[: i + 1]).item() for i in range(self.n_data)]

    def __len__(self):
        """Such that we can use len(data_handler)."""
        return self.n_data

    @staticmethod
    def _validate_data_args(x: Tensor, y: Tensor):
        """Checks if the dimensions of the training data is correct."""
        if x.dim() != 2:
            message = f"Input dimension is assumed 2-dim. not {x.ndim}-dim."
            raise BotorchTensorDimensionError(message)
        if y.dim() != 2:
            message = f"Output dimension is assumed 2-dim. not {x.ndim}-dim."
            raise BotorchTensorDimensionError(message)
        if y.shape[1] != 1:
            message = "We only support 1-dimensional outputs for the moment."
            raise BotorchTensorDimensionError(message)
        if x.shape[0] != y.shape[0]:
            message = "Not the number of input/ouput data."
            raise BotorchTensorDimensionError(message)
