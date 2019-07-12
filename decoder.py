import torch


class DEC(torch.nn.Module):
    def __init__(self, out_ch, in_ch, nf, ks):
        super(DEC, self).__init__()
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()

        self.conv1 = torch.nn.Conv2d(in_ch, nf, ks, padding=int((ks - 1)*0.5))
        self.bn1 = torch.nn.BatchNorm2d(nf)
        self.conv2 = torch.nn.Conv2d(nf, out_ch, ks, padding=int((ks - 1)*0.5))


    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        out = self.tanh(self.conv2(h))
        return out