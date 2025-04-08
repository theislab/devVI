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
    hidden_dims
        List of hidden dimensions
    latent_dim
        Latent dimension
    dropout
        Dropout rate
    """

    model: nn.Sequential

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        super().__init__()

        layers = []

        if not hidden_dims:
            # If no hidden dimensions, create direct connection
            layers.append(nn.Linear(input_dim, latent_dim))
        else:
            # Input layer
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )

            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.extend(
                    [
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout),
                    ]
                )

            # Output layer
            layers.append(nn.Linear(hidden_dims[-1], latent_dim))

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using Kaiming initialization.

        Parameters
        ----------
        m
            Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        return self.model(x)


class Decoder(nn.Module):
    """The decoder module.

    Parameters
    ----------
    latent_dim
        Latent dimension
    hidden_dims
        List of hidden dimensions (in reverse order of encoder)
    output_dim
        Output dimension
    dropout
        Dropout rate
    """

    model: nn.Sequential

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        output_dim: int = 784,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        layers = []

        if not hidden_dims:
            # If no hidden dimensions, create direct connection with sigmoid
            layers.extend([nn.Linear(latent_dim, output_dim), nn.Sigmoid()])
        else:
            # Input layer
            layers.extend(
                [
                    nn.Linear(latent_dim, hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )

            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                layers.extend(
                    [
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        nn.LeakyReLU(0.2),
                        nn.Dropout(dropout),
                    ]
                )

            # Output layer
            layers.extend([nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid()])

        self.model = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using Kaiming initialization.

        Parameters
        ----------
        m
            Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        return self.model(x)


class AutoEncoder(L.LightningModule):
    """PyTorch Lightning Autoencoder model.

    Parameters
    ----------
    input_dim
        Input dimension
    hidden_dims
        List of hidden dimensions for encoder (reversed for decoder)
    latent_dim
        Latent dimension
    dropout
        Dropout rate
    learning_rate
        Initial learning rate
    weight_decay
        Weight decay for AdamW optimizer
    scheduler_factor
        Factor by which to reduce learning rate on plateau
    scheduler_patience
        Number of epochs to wait before reducing learning rate
    """

    encoder: Encoder
    decoder: Decoder
    learning_rate: float
    weight_decay: float
    scheduler_factor: float
    scheduler_patience: int

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        # Initialize encoder and decoder
        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims, latent_dim=latent_dim, dropout=dropout)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
            output_dim=input_dim,
            dropout=dropout,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

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
        """Training step with L1 regularization.

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
        x_hat, z = self(x)

        # MSE reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # L1 regularization on latent space
        l1_reg = 1e-4 * torch.norm(z, p=1, dim=1).mean()

        # Total loss
        loss = recon_loss + l1_reg

        # Log losses
        self.log("train_loss", loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_l1_reg", l1_reg)

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
        x_hat, z = self(x)

        # MSE reconstruction loss
        recon_loss = F.mse_loss(x_hat, x)

        # L1 regularization
        l1_reg = 1e-4 * torch.norm(z, p=1, dim=1).mean()

        # Total loss
        loss = recon_loss + l1_reg

        # Log losses
        self.log("val_loss", loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_l1_reg", l1_reg)

    def configure_optimizers(self):
        """Configure optimizers and learning rate scheduler.

        Returns
        -------
        Dict with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=self.scheduler_factor, patience=self.scheduler_patience
        )

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
