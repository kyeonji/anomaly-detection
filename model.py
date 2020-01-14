import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self,):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(440, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, 40)
        self.fc32 = nn.Linear(512, 40)
        self.fc4 = nn.Linear(40, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 440)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 440))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar