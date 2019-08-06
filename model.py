import torch

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class Upsample(torch.nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)


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



class AE(torch.nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.hidden_size = 64
        self.nf = 64
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.enc_conv1 = torch.nn.Conv2d(1, self.nf, 3, stride=2, padding=1)
        self.enc_bn1 = torch.nn.BatchNorm2d(self.nf)
        self.enc_conv2 = torch.nn.Conv2d(self.nf, self.nf * 2, 3, stride=2, padding=1)
        self.enc_bn2 = torch.nn.BatchNorm2d(self.nf * 2)
        self.enc_conv3 = torch.nn.Conv2d(self.nf * 2, self.nf * 4, 3, stride=2, padding=1)
        self.enc_bn3 = torch.nn.BatchNorm2d(self.nf * 4)
        self.enc_conv4 = torch.nn.Conv2d(self.nf * 4, self.nf * 8, 3, stride=2, padding=1)
        self.enc_bn4 = torch.nn.BatchNorm2d(self.nf * 8)
        self.enc_conv5 = torch.nn.Conv2d(self.nf * 8, self.nf * 16, 3, stride=2, padding=1)
        self.enc_bn5 = torch.nn.BatchNorm2d(self.nf * 16)
        self.enc_conv6 = torch.nn.Conv2d(self.nf * 16, self.nf * 32, 3, stride=2, padding=1)

        self.upsample1 = Upsample(2)
        self.dec_conv1 = torch.nn.Conv2d(self.nf * 32, self.nf * 16, kernel_size=3, stride=1, padding=1)
        self.dec_bn1 = torch.nn.BatchNorm2d(self.nf * 16)
        self.upsample2 = Upsample(2)
        self.dec_conv2 = torch.nn.Conv2d(self.nf * 16, self.nf * 8, kernel_size=3, stride=1, padding=1)
        self.dec_bn2 = torch.nn.BatchNorm2d(self.nf * 8)
        self.upsample3 = Upsample(2)
        self.dec_conv3 = torch.nn.Conv2d(self.nf * 8, self.nf * 4, kernel_size=3, stride=1, padding=1)
        self.dec_bn3 = torch.nn.BatchNorm2d(self.nf * 4)
        self.upsample4 = Upsample(2)
        self.dec_conv4 = torch.nn.Conv2d(self.nf * 4, self.nf * 2, kernel_size=3, stride=1, padding=1)
        self.dec_bn4 = torch.nn.BatchNorm2d(self.nf * 2)
        self.upsample5 = Upsample(2)
        self.dec_conv5 = torch.nn.Conv2d(self.nf * 2, self.nf, kernel_size=3, stride=1, padding=1)
        self.dec_bn5 = torch.nn.BatchNorm2d(self.nf)
        self.upsample6 = Upsample(2)
        self.dec_conv6 = torch.nn.Conv2d(self.nf, 2, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x = self.relu(self.enc_bn1(self.enc_conv1(x)))
        x = self.relu(self.enc_bn2(self.enc_conv2(x)))
        x = self.relu(self.enc_bn3(self.enc_conv3(x)))
        x = self.relu(self.enc_bn4(self.enc_conv4(x)))
        x = self.relu(self.enc_bn5(self.enc_conv5(x)))
        x = self.enc_conv6(x)
        return x

    def decode(self, x):
        x = self.relu(self.dec_bn1(self.dec_conv1(self.upsample1(x))))
        x = self.relu(self.dec_bn2(self.dec_conv2(self.upsample2(x))))
        x = self.relu(self.dec_bn3(self.dec_conv3(self.upsample3(x))))
        x = self.relu(self.dec_bn4(self.dec_conv4(self.upsample4(x))))
        x = self.relu(self.dec_bn5(self.dec_conv5(self.upsample5(x))))
        x = self.dec_conv6(self.upsample6(x))
        return x

    def forward(self, lab):
        latent = self.encode(lab)
        lab_out = self.decode(latent)
        return lab_out, latent


def sanity_check(model, x, y, lr=2e-4, wd=0., epochs=20, device="cpu"):
    model.to(device)
    model.train()
    mse = torch.nn.MSELoss(reduction="none").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for i in range(epochs):
        output = model(x)
        # loss = mse(output, x).sum(-1).sum(-1).sum(-1).mean()
        loss = mse(output, y).sum(-1).sum(-1).sum(-1).mean()
        loss.backward()
        optimizer.step()
        print("Epoch {} loss {}".format(i, loss.item()))
    return



if __name__ == "__main__":
    from utils import *
    vae = AE()
    print("{:.0f}M".format(count_parameters(vae) * 1e-6))
    n = 10
    h, w = [64, 64]
    lab = torch.randn((n, 2, h, w))
    g = torch.randn((n, 1, h, w))
    sanity_check(vae, g, lab, lr=2e-5, epochs=10, wd=1e-6)
    vae.eval()
    lab_out = vae(g)
