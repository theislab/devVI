# import lightning as L
import torch

# import torch.nn.functional as F
# from torch.utils.data import DataLoader
from torch import nn


def basic_plot(adata: nn.Module) -> int:
    """Generate a basic plot for an AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Import matplotlib and implement a plotting function here.")
    return 0


class Encoder(nn.Module):
    """The encoder module.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        adata
            The AnnData object to preprocess.

        Returns
        -------
        Some integer value.
        """
        return self.l1(x)


class Decoder(nn.Module):
    """The decoder module.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    """

    def __init__(self):
        super().__init__()
        self.tensor = torch.Tensor([1, 2, 3])

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        adata
            The AnnData object to preprocess.

        Returns
        -------
        Some integer value.
        """
        return x


class BasicClass:
    """A basic class.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    """

    my_attribute: str = "Some attribute."
    my_other_attribute: int = 0

    def __init__(self, adata: nn.Module):
        print("Implement a class here.")

    def my_method(self, param: int) -> int:
        """A basic method.

        Parameters
        ----------
        param
            A parameter.

        Returns
        -------
        Some integer value.
        """
        print("Implement a method here.")
        return param + 5

    def my_other_method(self, param: str) -> str:
        """Another basic method.

        Parameters
        ----------
        param
            A parameter.

        Returns
        -------
        Some integer value.
        """
        pass
        return "Test"
