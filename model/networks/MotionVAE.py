import torch
from torch import nn
import numpy as np
from typing import List
from .HarmonicEmbedding import HarmonicEmbedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  # T*D
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # T*1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(
        self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu"
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.boneFeatQuery = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.muQuery = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.skelEmbedding = nn.Linear(self.nfeats, self.latent_dim)
        self.bone_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        boneTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.boneTransEncoder = nn.TransformerEncoder(boneTransEncoderLayer, num_layers=self.num_layers)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((1, 0, 3, 2)).reshape(njoints, bs * nframes, nfeats)
        x = self.skelEmbedding(x)
        xbone = torch.cat((self.boneFeatQuery.repeat(1, bs * nframes, 1), x), axis=0)
        xbone = self.boneTransEncoder(xbone)[0]  # bs*nframes, latent_dim
        xbone = xbone.reshape(bs, nframes, self.latent_dim).permute(1, 0, 2)
        xseq = torch.cat((self.muQuery.repeat(1, bs, 1), self.sigmaQuery.repeat(1, bs, 1), xbone), axis=0)
        xseq = self.sequence_pos_encoder(xseq)
        final = self.seqTransEncoder(xseq)
        mu = final[0]
        logvar = final[1]
        return [mu, logvar]


class Decoder(nn.Module):
    def __init__(
        self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation
        )
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        self.bone_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        boneTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation
        )
        self.boneTransDecoder = nn.TransformerDecoder(boneTransDecoderLayer, num_layers=self.num_layers)
        self.finallayer = nn.Linear(self.latent_dim, self.nfeats)

    def forward(self, z, nframes=1):
        if z.dim() == 2:
            z = z[None]
        ntoken, bs, latent_dim = z.shape
        njoints, nfeats = self.njoints, self.nfeats
        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        timequeries = self.sequence_pos_encoder(timequeries)
        if hasattr(self, 'q_weight'):
            timequeries = timequeries * self.q_weight
        if hasattr(self, 'z_weight'):
            z = z * self.z_weight
        seq_feature = self.seqTransDecoder(tgt=timequeries, memory=z)
        seq_feature = seq_feature.reshape(nframes * bs, latent_dim)[None]
        bonequeries = torch.zeros(njoints, nframes * bs, latent_dim, device=z.device)
        bonequeries = self.bone_pos_encoder(bonequeries)
        seq_bone_feature = self.boneTransDecoder(tgt=bonequeries, memory=seq_feature)
        seq_bone_feature = seq_bone_feature.reshape(njoints, nframes, bs, latent_dim)
        output = self.finallayer(seq_bone_feature).permute(2, 0, 3, 1)
        return output


class ArticulationVAE(nn.Module):
    def __init__(
        self, njoints, feat_dim, pos_dim, n_harmonic_functions=0, harmonic_omega0=1, latent_dim=256, z_token_num=10,
        pe_dropout=0, transformer_layer_num=4
    ) -> None:
        super(ArticulationVAE, self).__init__()
        self.njoints = njoints
        self.nfeats = feat_dim + pos_dim * (n_harmonic_functions * 2 + 1)
        self.latent_dim = latent_dim
        self.z_token_num = z_token_num
        self.transformer_layer_num = transformer_layer_num
        self.posenc = HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions, scalar=harmonic_omega0)
        self.in_layer = nn.Sequential(
            nn.Linear(self.nfeats, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.encoder = Encoder(
            njoints, latent_dim, latent_dim, dropout=pe_dropout, num_layers=self.transformer_layer_num
        )
        self.decoder = Decoder(
            njoints, 3, latent_dim, dropout=pe_dropout, num_layers=self.transformer_layer_num
        )

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        return self.encoder(input)

    def decode(self, z: torch.Tensor, nframes=1) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        return self.decoder(z, nframes)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        B, D = mu.shape
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.z_token_num, B, D).to(mu.device)
        return eps * std[None] + mu[None]

    def forward(self, inputs: torch.Tensor, pos: torch.Tensor, nframes, batch_size) -> List[torch.Tensor]:
        pos = torch.cat([pos, self.posenc(pos)], dim=-1)
        inputs = torch.cat([inputs, pos], dim=-1)
        inputs = self.in_layer(inputs)
        inputs = inputs.reshape(batch_size, nframes, *inputs.shape[1:])  # B, F, J, D
        inputs = inputs.permute(0, 2, 3, 1)  # B, J, D, F
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z, nframes).permute(0, 3, 1, 2).contiguous(), inputs, mu, log_var]

    def sample(self, num_sequence=1, num_frames=10) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param sequence_num: (Int) Number of sequence
        :param frame_per_sequence: (Int) Number of frames per sequence
        :param bone_num: (Int) Number of bones
        :return: (Tensor)
        """
        device = self.encoder.skelEmbedding.weight.device
        if hasattr(self, 'seed') and self.seed is not None:
            torch.manual_seed(self.seed)
        z = torch.randn(self.z_token_num, num_sequence, self.latent_dim)
        z = z.to(device) * 1.5
        samples = self.decode(z, num_frames).permute(0, 3, 1, 2).contiguous()
        return samples

    def forward_decoder(self, z, frame_per_sequence=8, **kwargs) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param sequence_num: (Int) Number of sequence
        :param frame_per_sequence: (Int) Number of frames per sequence
        :param bone_num: (Int) Number of bones
        :return: (Tensor)
        """
        samples = self.decode(z, frame_per_sequence).permute(0, 3, 1, 2).contiguous()
        return samples
