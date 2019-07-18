import torch


class DEC(torch.nn.Module):
    def __init__(self, out_ch, in_ch, nf, nlayers=3, ks=3):
        super(DEC, self).__init__()
        m = int(nlayers * 0.5)
        nfilters = torch.zeros(nlayers, dtype=torch.int)
        for i in range(m + 1):
            nfilters[i] = 2 ** i
            nfilters[- 1 - i] = 2 ** i

        layers = []
        for i in range(nlayers):
            if i == 0:
                layers.append(torch.nn.Conv2d(in_ch, nfilters[i].item() * nf, ks, padding=int((ks - 1)*0.5)))
                layers.append(torch.nn.BatchNorm2d(nfilters[i].item() * nf))
                layers.append(torch.nn.ReLU())
            elif i == nlayers - 1:
                layers.append(torch.nn.Conv2d(nfilters[i - 1].item() * nf, out_ch, ks, padding=int((ks - 1) * 0.5)))
                layers.append(torch.nn.Tanh())
            else:
                layers.append(torch.nn.Conv2d(nfilters[i - 1].item() * nf, nfilters[i].item() * nf, ks, padding=int((ks - 1) * 0.5)))
                layers.append(torch.nn.BatchNorm2d(nfilters[i].item() * nf))
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)


    def forward(self, x):
        out = self.net(x)
        return out


class AE(torch.nn.Module):
    def __init__(self, out_ch, in_ch, nf, ks=3):
        super(AE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, nf, ks, stride=2, padding=int(ks / 2)),
            torch.nn.BatchNorm2d(nf),
            torch.nn.ReLU(),
            torch.nn.Conv2d(nf, int(nf * 2), ks, stride=2, padding=int(ks / 2)),
            torch.nn.BatchNorm2d(int(nf * 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(int(nf * 2), int(nf * 4), ks, stride=2, padding=int(ks / 2)),
            torch.nn.BatchNorm2d(int(nf * 4)),
            torch.nn.ReLU()
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(int(nf * 4), int(nf * 2), kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(int(nf * 2)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(int(nf * 2), int(nf), kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(int(nf)),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(int(nf), int(out_ch), kernel_size=2, stride=2),
            torch.nn.Tanh()
        )

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out
