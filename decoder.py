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