import torch

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)


class UnFlatten(torch.nn.Module):
    def __init__(self, ch, s):
        super(UnFlatten, self).__init__()
        self.ch = ch
        self.s = s

    def forward(self, input):
        return input.reshape(input.shape[0], self.ch, self.s, self.s)


class CONVAE(torch.nn.Module):
    def __init__(self, inch, nch, ks, ld):
        super(CONVAE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(inch, nch, ks, stride=2, padding=int(ks/2)),
            torch.nn.BatchNorm2d(nch),
            torch.nn.ReLU(),
            torch.nn.Conv2d(nch, int(nch * 2), ks, stride=2, padding=int(ks/2)),
            torch.nn.BatchNorm2d(int(nch * 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(nch * 2), int(nch * 4), ks, stride=2, padding=int(ks/2)),
            torch.nn.BatchNorm2d(int(nch * 4)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(nch * 4), int(nch * 8), ks, stride=2, padding=int(ks/2)),
            torch.nn.BatchNorm2d(int(nch * 8)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(nch * 8), ld, ks, stride=2, padding=int(ks / 2))
            #Flatten(),
            #torch.nn.Linear(int(nch * 8) * 2 * int(32 / (2**4)), ld)
            )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(ld, int(nch * 8), kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(int(nch * 8)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(int(nch * 8), int(nch * 4), kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(int(nch * 4)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(int(nch * 4), int(nch * 2), kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(int(nch * 2)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(int(nch * 2), nch, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(nch),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(nch, inch, kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(inch),
            torch.nn.Sigmoid()
        )
        """
        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2.0, mode="bilinear"),
            torch.nn.Conv2d(ld, int(nch * 8), kernel_size=ks, padding=int(ks/2)),
            torch.nn.BatchNorm2d(int(nch * 8)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2.0, mode="bilinear"),
            torch.nn.Conv2d(int(nch * 8), int(nch * 4), kernel_size=ks, padding=int(ks/2)),
            torch.nn.BatchNorm2d(int(nch * 4)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2.0, mode="bilinear"),
            torch.nn.Conv2d(int(nch * 4), int(nch * 2), kernel_size=ks, padding=int(ks/2)),
            torch.nn.BatchNorm2d(int(nch * 2)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2.0, mode="bilinear"),
            torch.nn.Conv2d(int(nch * 2), inch, kernel_size=ks, padding=int(ks/2)),
            torch.nn.BatchNorm2d(inch),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2.0, mode="bilinear"),
            torch.nn.Sigmoid()
        )
        """


    def encode(self, x):
        return self.encoder(x)

    def decode(self, y):
        return self.decoder(y)

    def forward(self, x):
        h = self.encode(x).squeeze()
        ab = self.decode(h.reshape(h.shape[0], h.shape[1], 1, 1))
        return ab, h
