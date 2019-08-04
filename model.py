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


class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.hidden_size = 64
        self.nf = 64
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.enc_conv1 = torch.nn.Conv2d(2, self.nf, 5, stride=2, padding=2)
        self.enc_bn1 = torch.nn.BatchNorm2d(self.nf)
        self.enc_conv2 = torch.nn.Conv2d(self.nf, self.nf * 2, 5, stride=2, padding=2)
        self.enc_bn2 = torch.nn.BatchNorm2d(self.nf * 2)
        self.enc_conv3 = torch.nn.Conv2d(self.nf * 2, self.nf * 4, 5, stride=2, padding=2)
        self.enc_bn3 = torch.nn.BatchNorm2d(self.nf * 4)
        self.enc_conv4 = torch.nn.Conv2d(self.nf * 4, self.nf * 8, 3, stride=2, padding=1)
        self.enc_bn4 = torch.nn.BatchNorm2d(self.nf * 8)
        self.enc_conv5 = torch.nn.Conv2d(self.nf * 8, self.nf * 16, 3, stride=2, padding=1)
        self.enc_bn5 = torch.nn.BatchNorm2d(self.nf * 16)
        self.enc_conv_mu = torch.nn.Conv2d(self.nf * 16, self.hidden_size, 3, stride=2, padding=1)
        self.enc_conv_logvar = torch.nn.Conv2d(self.nf * 16, self.hidden_size, 3, stride=2, padding=1)

        self.dec_upsamp1 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = torch.nn.Conv2d(self.hidden_size, self.nf * 16, 3, stride=1, padding=1)
        self.dec_bn1 = torch.nn.BatchNorm2d(self.nf * 16)
        self.dec_upsamp2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = torch.nn.Conv2d(self.nf * 16, self.nf * 8, 5, stride=1, padding=2)
        self.dec_bn2 = torch.nn.BatchNorm2d(self.nf * 8)
        self.dec_upsamp3 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = torch.nn.Conv2d(self.nf * 8, self.nf * 4, 5, stride=1, padding=2)
        self.dec_bn3 = torch.nn.BatchNorm2d(self.nf * 4)
        self.dec_upsamp4 =torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv4 = torch.nn.Conv2d(self.nf * 4, self.nf * 2, 5, stride=1, padding=2)
        self.dec_bn4 = torch.nn.BatchNorm2d(self.nf * 2)
        self.dec_upsamp5 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv5 = torch.nn.Conv2d(self.nf * 2, self.nf, 5, stride=1, padding=2)
        self.dec_bn5 = torch.nn.BatchNorm2d(self.nf)
        self.dec_upsamp6 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv6 = torch.nn.Conv2d(self.nf, 2, 5, stride=1, padding=2)

    def encoder(self, x):
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))
        mu = self.enc_conv_mu(x)
        logvar =  self.tanh(self.enc_conv_logvar(x))
        return mu, logvar

    def decoder(self, z):
        x = self.relu(self.dec_bn1(self.dec_conv1(self.dec_upsamp1(z))))
        x = self.relu(self.dec_bn2(self.dec_conv2(self.dec_upsamp2(x))))
        x = self.relu(self.dec_bn3(self.dec_conv3(self.dec_upsamp3(x))))
        x = self.relu(self.dec_bn4(self.dec_conv4(self.dec_upsamp4(x))))
        x = self.relu(self.dec_bn5(self.dec_conv5(self.dec_upsamp5(x))))
        x = self.tanh(self.dec_conv6(self.dec_upsamp6(x)))
        return x

    def forward(self, color):
        mu, logvar = self.encoder(color)
        stddev = torch.sqrt(torch.exp(logvar))
        sample = torch.randn(stddev.shape, device=stddev.device)
        z = torch.add(mu, torch.mul(sample, stddev))
        lab_out = self.decoder(z)
        return mu.squeeze(), logvar.squeeze(), lab_out


class MDN(torch.nn.Module):
    def __init__(self):
        super(MDN, self).__init__()
        self.hidden_size = 64
        self.nf = 64
        self.relu = torch.nn.ReLU()

        self.enc_conv1 = torch.nn.Conv2d(1, self.nf, 5, stride=2, padding=2)
        self.enc_bn1 = torch.nn.BatchNorm2d(self.nf)
        self.enc_conv2 = torch.nn.Conv2d(self.nf, self.nf * 2, 5, stride=2, padding=2)
        self.enc_bn2 = torch.nn.BatchNorm2d(self.nf * 2)
        self.enc_conv3 = torch.nn.Conv2d(self.nf * 2, self.nf * 4, 5, stride=2, padding=2)
        self.enc_bn3 = torch.nn.BatchNorm2d(self.nf * 4)
        self.enc_conv4 = torch.nn.Conv2d(self.nf * 4, self.nf * 8, 3, stride=2, padding=1)
        self.enc_bn4 = torch.nn.BatchNorm2d(self.nf * 8)
        self.enc_conv5 = torch.nn.Conv2d(self.nf * 8, self.nf * 16, 3, stride=2, padding=1)
        self.enc_bn5 = torch.nn.BatchNorm2d(self.nf * 16)
        self.enc_conv_mu = torch.nn.Conv2d(self.nf * 16, self.hidden_size, 3, stride=2, padding=1)

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))
        x = self.enc_conv_mu(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        return z.squeeze()


if __name__ == "__main__":
    from utils import *
    vae = VAE()
    mdn = MDN()
    print("{:.0f}M".format(count_parameters(vae) * 1e-6))
    print("{:.0f}M".format(count_parameters(mdn) * 1e-6))
    n = 10
    h, w = [64, 64]
    lab = torch.randn((n, 2, h, w))
    g = torch.randn((n, 1, h, w))
    vae.eval()
    mu, logvar, lab_out = vae(lab)
    mu_mdn = mdn(g)
