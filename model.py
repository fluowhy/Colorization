import torch


class CONVAE(torch.nn.Module):
    def __init__(self):
        super(CONVAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 5),
            torch.nn.ReLU()
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, 5),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 2, 5),
            torch.nn.ReLU()
            )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        h = self.encode(x)
        ab = self.decode(h)
        return ab, h
