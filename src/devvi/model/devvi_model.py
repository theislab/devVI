import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """The encoder module.

    Parameters
    ----------
    input_dim
        Input dimension
    hidden_dim
        Hidden dimension
    latent_dim
        Latent dimension
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 64, latent_dim: int = 3):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
        Encoded representation
        """
        return self.l1(x)


class Decoder(nn.Module):
    """The decoder module.

    Parameters
    ----------
    latent_dim
        Latent dimension
    hidden_dim
        Hidden dimension
    output_dim
        Output dimension
    """

    def __init__(self, latent_dim: int = 3, hidden_dim: int = 64, output_dim: int = 784):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
        Reconstructed tensor
        """
        return self.l1(x)


class AutoEncoder(L.LightningModule):
    """PyTorch Lightning Autoencoder model.

    Parameters
    ----------
    input_dim
        Input dimension
    hidden_dim
        Hidden dimension
    latent_dim
        Latent dimension
    learning_rate
        Learning rate for optimization
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 64,
        latent_dim: int = 3,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

        self.learning_rate = learning_rate

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x
            Input tensor

        Returns
        -------
        Tuple of (reconstruction, latent)
        """
        z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z

    def training_step(self, batch, batch_idx):
        """Training step.

        Parameters
        ----------
        batch
            Input batch
        batch_idx
            Batch index

        Returns
        -------
        Loss value
        """
        x = batch
        x_hat, _ = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Parameters
        ----------
        batch
            Input batch
        batch_idx
            Batch index
        """
        x = batch
        x_hat, _ = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        """Configure optimizers.

        Returns
        -------
        Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


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
