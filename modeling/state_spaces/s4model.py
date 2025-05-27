import torch.nn as nn

from modeling.state_spaces.models.s4.s4d import S4D


class S4Model_All(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_layers, dropout=0.2, prenorm=False):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=0.001))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)

        return x


class S4Model_Next(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_layers, dropout=0.2, prenorm=False):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=0.001))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=-2)  # (B, L, d_model) -> (B, d_model)
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


class S4Model_Extractor(nn.Module):
    def __init__(self, d_input, d_mid1, d_mid2, d_output, d_model, n_layers, dropout=0.2, prenorm=False):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=0.001))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_mid1),
            nn.ReLU(True),
            nn.Linear(d_mid1, d_mid2),
            nn.ReLU(True),
            nn.Linear(d_mid2, d_output)
        )

    def forward(self, x):
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        # Pooling: average pooling over the sequence length
        x = x.mean(dim=-2)
        x = self.decoder(x)  # (B, L, d_model) -> (B, d_output)

        return x
