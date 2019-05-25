import torch


class CNNAE(torch.nn.Module):
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
        """
        encoder_layers.append(torch.nn.Conv2d(inchannels, nch, kernel_size=5, stride=2))
        encoder_layers.append(torch.nn.BatchNorm2d(num_features=nch))
        encoder_layers.append(torch.nn.ReLU())
        encoder_layers.append(torch.nn.Conv2d(nch, int(nch*2), kernel_size=5, stride=2))
        encoder_layers.append(torch.nn.BatchNorm2d(num_features=int(nch*2)))
        encoder_layers.append(torch.nn.ReLU())
        encoder_layers.append(torch.nn.Conv2d(int(nch*2), int(nch*4), kernel_size=5, stride=2))
        encoder_layers.append(torch.nn.BatchNorm2d(num_features=int(nch*4)))
        encoder_layers.append(torch.nn.ReLU())
        encoder_layers.append(torch.nn.Conv2d(int(nch * 4), int(nch * 8), kernel_size=5, stride=2))
        encoder_layers.append(torch.nn.BatchNorm2d(num_features=int(nch * 8)))
        encoder_layers.append(torch.nn.ReLU())
        encoder_layers.append(torch.nn.Conv2d(int(nch * 8), ndim_latent, kernel_size=1, stride=0))
        """

    def encode(self, x):
    	return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        h = self.encode(x)
        ab = self.decode(h)
        return ab, h
