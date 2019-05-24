import torch


class CNNAE(torch.nn.Module):
    def __init__(self, inchannels, nch, ndim_latent):
        super(CNNAE, self).__init__()
        encoder_layers = []
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

        self.encoder = torch.nn.Sequential(*encoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, u):
        return

    def forward(self, x):
        l = self.encode(x)
        return l