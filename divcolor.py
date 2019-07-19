# auhtor: aditya12agd5
# modifications: fluowhy
import torch


class VAE(torch.nn.Module):
  def __init__(self, nf, hs):
    super(VAE, self).__init__()
    self.hidden_size = hs  # 64
    self.nf = nf  # 128
    self.tanh = torch.nn.Tanh()
    self.relu = torch.nn.ReLU()

    #Encoder layers
    self.enc_conv1 = torch.nn.Conv2d(2, nf, 5, stride=2, padding=2)
    self.enc_bn1 = torch.nn.BatchNorm2d(nf)
    self.enc_conv2 = torch.nn.Conv2d(nf, 2 * nf, 5, stride=2, padding=2)
    self.enc_bn2 = torch.nn.BatchNorm2d(2 * nf)
    self.enc_conv3 = torch.nn.Conv2d(2 * nf, 4 * nf, 5, stride=2, padding=2)
    self.enc_bn3 = torch.nn.BatchNorm2d(4 * nf)
    self.enc_conv4 = torch.nn.Conv2d(4 * nf, 8 * nf, 3, stride=2, padding=1)
    self.enc_bn4 = torch.nn.BatchNorm2d(8 * nf)
    self.enc_fc1 = torch.nn.Linear(4 * 4 * 8 * nf, self.hidden_size * 2)
    self.enc_dropout1 = torch.nn.Dropout(p=.7)

    #Cond encoder layers
    self.cond_enc_conv1 = torch.nn.Conv2d(1, nf, 5, stride=2, padding=2)
    self.cond_enc_bn1 = torch.nn.BatchNorm2d(nf)
    self.cond_enc_conv2 = torch.nn.Conv2d(nf, 2 * nf, 5, stride=2, padding=2)
    self.cond_enc_bn2 = torch.nn.BatchNorm2d(2 * nf)
    self.cond_enc_conv3 = torch.nn.Conv2d(2 * nf, 4 * nf, 5, stride=2, padding=2)
    self.cond_enc_bn3 = torch.nn.BatchNorm2d(4 * nf)
    self.cond_enc_conv4 = torch.nn.Conv2d(4 * nf, 8 * nf, 3, stride=2, padding=1)
    self.cond_enc_bn4 = torch.nn.BatchNorm2d(8 * nf)

    #Decoder layers
    self.dec_upsamp1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')
    self.dec_conv1 = torch.nn.Conv2d(8 * nf + self.hidden_size, 4 * nf, 3, stride=1, padding=1)
    self.dec_bn1 = torch.nn.BatchNorm2d(4 * nf)
    self.dec_upsamp2 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv2 = torch.nn.Conv2d(4 * nf * 2, 2 * nf, 5, stride=1, padding=2)
    self.dec_bn2 = torch.nn.BatchNorm2d(2 * nf)
    self.dec_upsamp3 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv3 = torch.nn.Conv2d(2 * nf * 2, nf, 5, stride=1, padding=2)
    self.dec_bn3 = torch.nn.BatchNorm2d(nf)
    self.dec_upsamp4 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv4 = torch.nn.Conv2d(nf * 2, int(nf * 0.5), 5, stride=1, padding=2)
    self.dec_bn4 = torch.nn.BatchNorm2d(int(nf * 0.5))
    self.dec_upsamp5 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv5 = torch.nn.Conv2d(int(nf * 0.5), 2, 5, stride=1, padding=2)

  def encoder(self, x):
    x = self.relu(self.enc_conv1(x))
    x = self.enc_bn1(x)
    x = self.relu(self.enc_conv2(x))
    x = self.enc_bn2(x)
    x = self.relu(self.enc_conv3(x))
    x = self.enc_bn3(x)
    x = self.relu(self.enc_conv4(x))
    x = self.enc_bn4(x)
    x = x.view(-1, 4 * 4 * self.nf * 8)
    x = self.enc_dropout1(x)
    x = self.enc_fc1(x)
    mu = x[..., :self.hidden_size]
    logvar = x[..., self.hidden_size:]
    return mu, logvar

  def cond_encoder(self, x):
    x = self.relu(self.cond_enc_conv1(x))
    sc_feat32 = self.cond_enc_bn1(x)
    x = self.relu(self.cond_enc_conv2(sc_feat32))
    sc_feat16 = self.cond_enc_bn2(x)
    x = self.relu(self.cond_enc_conv3(sc_feat16))
    sc_feat8 = self.cond_enc_bn3(x)
    x = self.relu(self.cond_enc_conv4(sc_feat8))
    sc_feat4 = self.cond_enc_bn4(x)
    return sc_feat32, sc_feat16, sc_feat8, sc_feat4

  def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):
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
      
  def forward(self, color, greylevel):
    sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
    mu, logvar = self.encoder(color)
    stddev = torch.sqrt(torch.exp(logvar))
    sample = torch.randn(stddev.shape, device=stddev.device)
    z = torch.add(mu, torch.mul(sample, stddev))
    color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
    return mu, logvar, color_out

class MDN(torch.nn.Module):
  def __init__(self, nf, hs):
    super(MDN, self).__init__()
    self.hidden_size = hs
    self.nf = nf
    self.relu = torch.nn.ReLU()

    self.mdn_conv1 = torch.nn.Conv2d(1, nf, 5, stride=1, padding=2)
    self.mdn_bn1 = torch.nn.BatchNorm2d(nf)
    self.mdn_conv2 = torch.nn.Conv2d(nf, 2 * nf, 5, stride=1, padding=2)
    self.mdn_bn2 = torch.nn.BatchNorm2d(2 * nf)
    self.mdn_conv3 = torch.nn.Conv2d(2 * nf, 4 * nf, 5, stride=1, padding=2)
    self.mdn_bn3 = torch.nn.BatchNorm2d(4 * nf)
    self.mdn_conv4 = torch.nn.Conv2d(4 * nf, 8 * nf, 5, stride=2, padding=2)
    self.mdn_bn4 = torch.nn.BatchNorm2d(8 * nf)
    self.mdn_conv5 = torch.nn.Conv2d(8 * nf, 4 * nf, 5, stride=1, padding=2)
    self.mdn_bn5 = torch.nn.BatchNorm2d(4 * nf)
    self.mdn_conv6 = torch.nn.Conv2d(4 * nf, 2 * nf, 5, stride=2, padding=2)
    self.mdn_bn6 = torch.nn.BatchNorm2d(2 * nf)
    self.mdn_conv7 = torch.nn.Conv2d(2 * nf, nf, 5, stride=2, padding=2)
    self.mdn_bn7 = torch.nn.BatchNorm2d(nf)
    self.mdn_dropout1 = torch.nn.Dropout(p=.7)
    self.mdn_fc1 = torch.nn.Linear(8 * 8 * nf, hs)

  def forward(self, grey):
    x = self.relu(self.mdn_conv1(grey))
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
    x = x.view(-1, 8 * 8 * self.nf)
    x = self.mdn_dropout1(x)
    x = self.mdn_fc1(x)
    return x