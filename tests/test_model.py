import pytest
import torch

from devvi.model.devvi_model import AutoEncoder, Decoder, Encoder


@pytest.fixture
def sample_data():
    batch_size = 32
    input_dim = 784
    return torch.randn(batch_size, input_dim)


def test_encoder_initialization():
    """Test encoder initialization with default and custom parameters."""
    # Test default initialization
    encoder = Encoder()
    assert encoder.model is not None

    # Test custom initialization
    custom_hidden_dims = [256, 128, 64]
    custom_latent_dim = 16
    custom_dropout = 0.2
    encoder = Encoder(
        input_dim=1000, hidden_dims=custom_hidden_dims, latent_dim=custom_latent_dim, dropout=custom_dropout
    )
    assert encoder.model is not None


def test_encoder_forward(sample_data):
    """Test encoder forward pass and output shapes."""
    batch_size = sample_data.shape[0]
    encoder = Encoder(input_dim=sample_data.shape[1], latent_dim=32)

    # Test forward pass
    latent = encoder(sample_data)
    assert latent.shape == (batch_size, 32)
    assert not torch.isnan(latent).any()


def test_decoder_initialization():
    """Test decoder initialization with default and custom parameters."""
    # Test default initialization
    decoder = Decoder()
    assert decoder.model is not None

    # Test custom initialization
    custom_hidden_dims = [64, 128, 256]
    custom_output_dim = 1000
    custom_dropout = 0.2
    decoder = Decoder(
        latent_dim=16, hidden_dims=custom_hidden_dims, output_dim=custom_output_dim, dropout=custom_dropout
    )
    assert decoder.model is not None


def test_decoder_forward():
    """Test decoder forward pass and output shapes."""
    batch_size = 32
    latent_dim = 32
    output_dim = 784

    decoder = Decoder(latent_dim=latent_dim, output_dim=output_dim)
    latent = torch.randn(batch_size, latent_dim)

    # Test forward pass
    reconstruction = decoder(latent)
    assert reconstruction.shape == (batch_size, output_dim)
    assert not torch.isnan(reconstruction).any()
    assert torch.all(reconstruction >= 0) and torch.all(reconstruction <= 1)  # Check sigmoid output bounds


def test_autoencoder_initialization():
    """Test autoencoder initialization with default and custom parameters."""
    # Test default initialization
    autoencoder = AutoEncoder()
    assert isinstance(autoencoder.encoder, Encoder)
    assert isinstance(autoencoder.decoder, Decoder)

    # Test custom initialization
    custom_hidden_dims = [256, 128, 64]
    custom_latent_dim = 16
    custom_dropout = 0.2
    custom_lr = 1e-4

    autoencoder = AutoEncoder(
        input_dim=1000,
        hidden_dims=custom_hidden_dims,
        latent_dim=custom_latent_dim,
        dropout=custom_dropout,
        learning_rate=custom_lr,
    )
    assert isinstance(autoencoder.encoder, Encoder)
    assert isinstance(autoencoder.decoder, Decoder)


def test_autoencoder_forward(sample_data):
    """Test autoencoder forward pass and output shapes."""
    batch_size = sample_data.shape[0]
    input_dim = sample_data.shape[1]
    latent_dim = 32

    autoencoder = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)

    # Test forward pass
    reconstruction, latent = autoencoder(sample_data)
    assert reconstruction.shape == sample_data.shape
    assert latent.shape == (batch_size, latent_dim)
    assert not torch.isnan(reconstruction).any()
    assert not torch.isnan(latent).any()


def test_autoencoder_training_step(sample_data):
    """Test autoencoder training step."""
    autoencoder = AutoEncoder(input_dim=sample_data.shape[1])

    # Test training step
    loss = autoencoder.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar loss
    assert not torch.isnan(loss)
    assert loss > 0  # Loss should be positive


def test_autoencoder_validation_step(sample_data):
    """Test autoencoder validation step."""
    autoencoder = AutoEncoder(input_dim=sample_data.shape[1])

    # Test validation step
    autoencoder.validation_step(sample_data, 0)
    # No assertion needed as validation_step doesn't return anything,
    # but we check it runs without errors


def test_autoencoder_configure_optimizers():
    """Test autoencoder optimizer configuration."""
    autoencoder = AutoEncoder()

    # Test optimizer configuration
    optim_config = autoencoder.configure_optimizers()
    assert isinstance(optim_config, dict)
    assert "optimizer" in optim_config
    assert "lr_scheduler" in optim_config
    assert isinstance(optim_config["optimizer"], torch.optim.AdamW)
    assert isinstance(optim_config["lr_scheduler"]["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_edge_cases():
    """Test edge cases and potential error conditions."""
    # Test with minimal dimensions
    min_encoder = Encoder(input_dim=1, hidden_dims=[2], latent_dim=1)
    min_decoder = Decoder(latent_dim=1, hidden_dims=[2], output_dim=1)

    # Set to eval mode for testing with single samples
    min_encoder.eval()
    min_decoder.eval()

    min_input = torch.randn(1, 1)
    min_latent = min_encoder(min_input)
    min_reconstruction = min_decoder(min_latent)
    assert min_reconstruction.shape == min_input.shape

    # Test with single sample
    single_input = torch.randn(1, 784)
    encoder = Encoder()
    decoder = Decoder()

    # Set to eval mode for testing with single samples
    encoder.eval()
    decoder.eval()

    latent = encoder(single_input)
    reconstruction = decoder(latent)
    assert reconstruction.shape == single_input.shape

    # Test with empty hidden_dims
    encoder_no_hidden = Encoder(hidden_dims=[])
    decoder_no_hidden = Decoder(hidden_dims=[])
    assert encoder_no_hidden.model is not None
    assert decoder_no_hidden.model is not None
