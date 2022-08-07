import torch
import torch.nn as nn
from chorale_phrase_tensor import GeneratedPhraseBatchTensor, PhraseTensor, cut_tensor_by_features


class PhraseEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, number_of_hidden_layers=2):
        super().__init__()

        modules = []

        layer_sizes = [in_dim]
        for i in range(number_of_hidden_layers):
            layer_sizes += [int((layer_sizes[-1] - out_dim) / 2) + out_dim]
        layer_sizes += [out_dim]

        for in_layer, out_layer in zip(layer_sizes[:-1], layer_sizes[1:]):
            modules.append(nn.Linear(in_layer, out_layer))
            modules.append(nn.ReLU())

        # Get rid of that last ReLU
        self.encoder = nn.Sequential(*modules[:-1])

    def forward(self, x):
        return self.encoder(x)


class PhraseDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, number_of_hidden_layers=2):
        super().__init__()

        modules = []

        # Calculate the layer sizes in reverse order so that it matches the sizes in the encoder
        layer_sizes = [out_dim]
        for i in range(number_of_hidden_layers):
            layer_sizes += [int((layer_sizes[-1] - in_dim) / 2) + in_dim]
        layer_sizes += [in_dim]

        layer_sizes.reverse()

        for in_layer, out_layer in zip(layer_sizes[:-1], layer_sizes[1:]):
            modules.append(nn.Linear(in_layer, out_layer))
            modules.append(nn.ReLU())

        # Get rid of that last ReLU
        self.decoder = nn.Sequential(*modules[:-1])

    def forward(self, x):
        return self.decoder(x)


class PhraseDecoderWithSoftmax(nn.Module):
    def __init__(self, in_dim, out_dim, number_of_hidden_layers=2):
        super().__init__()

        self.decoder = PhraseDecoder(in_dim, out_dim, number_of_hidden_layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        feature_scores = cut_tensor_by_features(self.decoder(x))
        softmax_scores = []
        for feature in feature_scores:
            softmax_scores.append(self.softmax(feature))
        return torch.concat([*softmax_scores], dim=1)


# This is copied from an exercise from Technion class 236781
class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder that extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        x0 = torch.rand((1, in_size)).to(next(features_encoder.parameters()).device)
        self.encoder_out_size = features_encoder(x0).shape[1]
        self.encoder_mean = nn.Linear(self.encoder_out_size, z_dim).to(next(features_encoder.parameters()).device)
        self.log_sigma2 = nn.Linear(self.encoder_out_size, z_dim).to(next(features_encoder.parameters()).device)

    # def _check_features(self, in_size):
    #     device = next(self.parameters()).device
    #     with torch.no_grad():
    #         # Make sure encoder and decoder are compatible
    #         x = torch.randn(1, *in_size, device=device)
    #         h = self.features_encoder(x)
    #         xr = self.features_decoder(h)
    #         assert xr.shape == x.shape
    #         # Return the shape and number of encoded features
    #         return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        out = self.features_encoder(x)
        mu = self.encoder_mean(out)
        log_sigma2 = self.log_sigma2(out)
        random = torch.randn((x.shape[0], self.z_dim)).to(next(self.log_sigma2.parameters()).device)
        z = torch.sqrt(torch.exp(log_sigma2)) * random + mu

        return z, mu, log_sigma2

    def decode(self, z):
        return self.features_decoder(z)

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(next(self.log_sigma2.parameters()).device)
            samples = self.features_decoder(z)

        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    reg = 1 / (x_sigma2 * x.shape[1])
    data_loss = reg * torch.norm((x - xr).reshape((x.shape[0], -1)), dim=1) ** 2.0
    kldiv_loss = (
        torch.sum(torch.exp(z_log_sigma2), dim=1) +
        torch.norm(z_mu, dim=1) ** 2.0 - z_mu.shape[1] -
        torch.sum(z_log_sigma2, dim=1)
    )

    data_loss = torch.mean(data_loss)
    kldiv_loss = torch.mean(kldiv_loss)
    loss = data_loss + kldiv_loss

    return loss, data_loss, kldiv_loss
