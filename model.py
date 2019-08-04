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

  # define layers
  def __init__(self):
    super(VAE, self).__init__()
    self.hidden_size = 64
    self.tanh = torch.nn.Tanh()
    self.relu = torch.nn.ReLU()

    # Encoder layers
    self.enc_conv1 = torch.nn.Conv2d(2, 128, 5, stride=2, padding=2)
    self.enc_bn1 = torch.nn.BatchNorm2d(128)
    self.enc_conv2 = torch.nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.enc_bn2 = torch.nn.BatchNorm2d(256)
    self.enc_conv3 = torch.nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.enc_bn3 = torch.nn.BatchNorm2d(512)
    self.enc_conv4 = torch.nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.enc_bn4 = torch.nn.BatchNorm2d(1024)
    self.enc_fc_mu = torch.nn.Linear(4 * 4 * 1024, self.hidden_size)
    self.enc_fc_logvar = torch.nn.Linear(4 * 4 * 1024, self.hidden_size)
    self.enc_dropout1 = torch.nn.Dropout(p=.7)

    # Decoder layers
    self.dec_upsamp1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    self.dec_conv1 = nn.Conv2d(1024 + self.hidden_size, 512, 3, stride=1, padding=1)
    self.dec_bn1 = nn.BatchNorm2d(512)
    self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv2 = nn.Conv2d(512, 256, 5, stride=1, padding=2)
    self.dec_bn2 = nn.BatchNorm2d(256)
    self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv3 = nn.Conv2d(256, 128, 5, stride=1, padding=2)
    self.dec_bn3 = nn.BatchNorm2d(128)
    self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv4 = nn.Conv2d(128, 64, 5, stride=1, padding=2)
    self.dec_bn4 = nn.BatchNorm2d(64)
    self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv5 = nn.Conv2d(64, 2, 5, stride=1, padding=2)

  def encoder(self, x):
    x = self.relu(self.enc_conv1(x))
    x = self.enc_bn1(x)
    x = self.relu(self.enc_conv2(x))
    x = self.enc_bn2(x)
    x = self.relu(self.enc_conv3(x))
    x = self.enc_bn3(x)
    x = self.relu(self.enc_conv4(x))
    x = self.enc_bn4(x)
    x = x.view(-1, 4 * 4 * 1024)
    x = self.enc_dropout1(x)
    mu = self.enc_fc_mu(x)
    logvar =  self.tanh(self.enc_fc_logvar(x))
    return mu, logvar

  def decoder(self, z):
    x = z.view(-1, self.hidden_size, 1, 1)
    x = self.dec_upsamp1(x)
    x = torch.cat([x, sc_feat4], 1)
    x = self.relu(self.dec_conv1(x))
    x = self.dec_bn1(x)
    x = self.dec_upsamp2(x)
    x = torch.cat([x, sc_feat8], 1)
    x = self.relu(self.dec_conv2(x))
    x = self.dec_bn2(x)
    x = self.dec_upsamp3(x)
    x = torch.cat([x, sc_feat16], 1)
    x = self.relu(self.dec_conv3(x))
    x = self.dec_bn3(x)
    x = self.dec_upsamp4(x)
    x = torch.cat([x, sc_feat32], 1)
    x = self.relu(self.dec_conv4(x))
    x = self.dec_bn4(x)
    x = self.dec_upsamp5(x)
    x = self.tanh(self.dec_conv5(x))
    return x

  # define forward pass
  def forward(self, color):
    mu, logvar = self.encoder(color)
    stddev = torch.sqrt(torch.exp(logvar))
    sample = torch.randn(stddev.shape, device=stddev.device)
    z = torch.add(mu, torch.mul(sample, stddev))
    color_out = self.decoder(z)
    return mu, logvar, color_out


class MDN(nn.Module):

  # define layers
  def __init__(self):
    super(MDN, self).__init__()
    self.hidden_size = 64
    self.relu = torch.nn.ReLU()

    # MDN Layers
    self.mdn_conv1 = nn.Conv2d(1, 512, 5, stride=2, padding=int(5 / 2))
    self.mdn_bn1 = nn.BatchNorm2d(512)
    self.mdn_conv2 = nn.Conv2d(512, 512, 5, stride=2, padding=int(5 / 2))
    self.mdn_bn2 = nn.BatchNorm2d(512)
    self.mdn_conv3 = nn.Conv2d(512, 288, 5, padding=int(5 / 2))
    self.mdn_bn3 = nn.BatchNorm2d(288)
    self.mdn_conv4 = nn.Conv2d(288, 256, 5, stride=2, padding=int(5 / 2))
    self.mdn_bn4 = nn.BatchNorm2d(256)
    self.mdn_conv5 = nn.Conv2d(256, 128, 5, padding=int(5 / 2))
    self.mdn_bn5 = nn.BatchNorm2d(128)
    self.mdn_conv6 = nn.Conv2d(128, 96, 5, stride=2, padding=int(5 / 2))
    self.mdn_bn6 = nn.BatchNorm2d(96)
    self.mdn_conv7 = nn.Conv2d(96, 64, 5, padding=int(5 / 2))
    self.mdn_bn7 = nn.BatchNorm2d(64)
    self.mdn_dropout1 = nn.Dropout(p=.7)
    self.mdn_fc1 = nn.Linear(4 * 4 * 64, self.hidden_size)

  # define forward pass
  def forward(self, x):
    x = self.relu(self.mdn_conv1(x))
    x = self.mdn_bn1(x)
    x = self.relu(self.mdn_conv2(x))
    x = self.mdn_bn2(x)
    x = self.relu(self.mdn_conv3(x))
    x = self.mdn_bn3(x)
    x = self.relu(self.mdn_conv4(x))
    x = self.mdn_bn4(x)
    x = self.relu(self.mdn_conv5(x))
    x = self.mdn_bn5(x)
    x = self.relu(self.mdn_conv6(x))
    x = self.mdn_bn6(x)
    x = self.relu(self.mdn_conv7(x))
    x = self.mdn_bn7(x)
    x = x.view(-1, 4 * 4 * 64)
    x = self.mdn_dropout1(x)
    x = self.mdn_fc1(x)
    return x