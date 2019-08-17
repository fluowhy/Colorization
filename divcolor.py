import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Upsample(torch.nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.scale_factor)


class VAE(nn.Module):
  
  #define layers
  def __init__(self):
    super(VAE, self).__init__()

    self.hidden_size = 64

    #Encoder layers
    self.enc_conv1 = nn.Conv2d(2, 128, 5, stride=2, padding=2)
    self.enc_bn1 = nn.BatchNorm2d(128)
    self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.enc_bn2 = nn.BatchNorm2d(256)
    self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.enc_bn3 = nn.BatchNorm2d(512)
    self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.enc_bn4 = nn.BatchNorm2d(1024)
    self.enc_fc1 = nn.Linear(4*4*1024, self.hidden_size*2)
    self.enc_dropout1 = nn.Dropout(p=.7)

    #Cond encoder layers
    self.cond_enc_conv1 = nn.Conv2d(1, 128, 5, stride=2, padding=2)
    self.cond_enc_bn1 = nn.BatchNorm2d(128)
    self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.cond_enc_bn2 = nn.BatchNorm2d(256)
    self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.cond_enc_bn3 = nn.BatchNorm2d(512)
    self.cond_enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.cond_enc_bn4 = nn.BatchNorm2d(1024)

    #Decoder layers
    self.dec_upsamp1 = nn.Upsample(scale_factor=4, mode='bilinear')
    self.dec_conv1 = nn.Conv2d(1024+self.hidden_size, 512, 3, stride=1, padding=1)
    self.dec_bn1 = nn.BatchNorm2d(512)
    self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv2 = nn.Conv2d(512*2, 256, 5, stride=1, padding=2)
    self.dec_bn2 = nn.BatchNorm2d(256)
    self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv3 = nn.Conv2d(256*2, 128, 5, stride=1, padding=2)
    self.dec_bn3 = nn.BatchNorm2d(128)
    self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv4 = nn.Conv2d(128*2, 64, 5, stride=1, padding=2)
    self.dec_bn4 = nn.BatchNorm2d(64)
    self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv5 = nn.Conv2d(64, 2, 5, stride=1, padding=2)

  def encoder(self, x):
    x = F.relu(self.enc_conv1(x))
    x = self.enc_bn1(x)
    x = F.relu(self.enc_conv2(x))
    x = self.enc_bn2(x)
    x = F.relu(self.enc_conv3(x))
    x = self.enc_bn3(x)
    x = F.relu(self.enc_conv4(x))
    x = self.enc_bn4(x)
    x = x.view(-1, 4*4*1024)
    x = self.enc_dropout1(x)
    x = self.enc_fc1(x)
    mu = x[..., :self.hidden_size]
    logvar = x[..., self.hidden_size:]
    return mu, logvar

  def cond_encoder(self, x):
    x = F.relu(self.cond_enc_conv1(x))
    sc_feat32 = self.cond_enc_bn1(x)
    x = F.relu(self.cond_enc_conv2(sc_feat32))
    sc_feat16 = self.cond_enc_bn2(x)
    x = F.relu(self.cond_enc_conv3(sc_feat16))
    sc_feat8 = self.cond_enc_bn3(x)
    x = F.relu(self.cond_enc_conv4(sc_feat8))
    sc_feat4 = self.cond_enc_bn4(x)
    return sc_feat32, sc_feat16, sc_feat8, sc_feat4

  def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):
    x = z.view(-1, self.hidden_size, 1, 1)
    x = self.dec_upsamp1(x)
    x = torch.cat([x, sc_feat4], 1)
    x = F.relu(self.dec_conv1(x))
    x = self.dec_bn1(x)
    x = self.dec_upsamp2(x) 
    x = torch.cat([x, sc_feat8], 1)
    x = F.relu(self.dec_conv2(x))
    x = self.dec_bn2(x)
    x = self.dec_upsamp3(x) 
    x = torch.cat([x, sc_feat16], 1)
    x = F.relu(self.dec_conv3(x))
    x = self.dec_bn3(x)
    x = self.dec_upsamp4(x) 
    x = torch.cat([x, sc_feat32], 1)
    x = F.relu(self.dec_conv4(x))
    x = self.dec_bn4(x)
    x = self.dec_upsamp5(x) 
    x = F.tanh(self.dec_conv5(x))
    return x
      
  #define forward pass
  def forward(self, color, greylevel, z_in, is_train=True):
    sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
    mu, logvar = self.encoder(color)
    if(is_train == True):
      stddev = torch.sqrt(torch.exp(logvar))
      eps = Variable(torch.randn(stddev.size()).normal_()).cuda()
      z = torch.add(mu, torch.mul(eps, stddev))
    else:
      z = z_in
    color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
    return mu, logvar, color_out


class MDN(nn.Module):
  
  #define layers
  def __init__(self):
    super(MDN, self).__init__()

    self.feats_nch = 512
    self.hidden_size = 64
    self.nmix = 8
    self.nout = (self.hidden_size+1)*self.nmix

    #MDN Layers
    self.mdn_conv1 = nn.Conv2d(self.feats_nch, 384, 5, stride=1, padding=2)
    self.mdn_bn1 = nn.BatchNorm2d(384)
    self.mdn_conv2 = nn.Conv2d(384, 320, 5, stride=1, padding=2)
    self.mdn_bn2 = nn.BatchNorm2d(320)
    self.mdn_conv3 = nn.Conv2d(320, 288, 5, stride=1, padding=2)
    self.mdn_bn3 = nn.BatchNorm2d(288)
    self.mdn_conv4 = nn.Conv2d(288, 256, 5, stride=2, padding=2)
    self.mdn_bn4 = nn.BatchNorm2d(256)
    self.mdn_conv5 = nn.Conv2d(256, 128, 5, stride=1, padding=2)
    self.mdn_bn5 = nn.BatchNorm2d(128)
    self.mdn_conv6 = nn.Conv2d(128, 96, 5, stride=2, padding=2)
    self.mdn_bn6 = nn.BatchNorm2d(96)
    self.mdn_conv7 = nn.Conv2d(96, 64, 5, stride=2, padding=2)
    self.mdn_bn7 = nn.BatchNorm2d(64)
    self.mdn_dropout1 = nn.Dropout(p=.7)
    self.mdn_fc1 = nn.Linear(4*4*64, self.nout)

  #define forward pass
  def forward(self, feats):
    x = F.relu(self.mdn_conv1(feats))
    x = self.mdn_bn1(x)
    x = F.relu(self.mdn_conv2(x))
    x = self.mdn_bn2(x)
    x = F.relu(self.mdn_conv3(x))
    x = self.mdn_bn3(x)
    x = F.relu(self.mdn_conv4(x))
    x = self.mdn_bn4(x)
    x = F.relu(self.mdn_conv5(x))
    x = self.mdn_bn5(x)
    x = F.relu(self.mdn_conv6(x))
    x = self.mdn_bn6(x)
    x = F.relu(self.mdn_conv7(x))
    x = self.mdn_bn7(x)
    x = x.view(-1, 4*4*64)
    x = self.mdn_dropout1(x)
    x = self.mdn_fc1(x)
    return x

class MDNModU(nn.Module):

  # define layers
  def __init__(self):
    super(MDNMod, self).__init__()
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

class VAEModU(nn.Module):

  # define layers
  def __init__(self):
    super(VAEMod, self).__init__()
    self.hidden_size = 64
    self.tanh = torch.nn.Tanh()
    self.relu = torch.nn.ReLU()

    # Encoder layers
    self.enc_conv1 = nn.Conv2d(2, 128, 5, stride=2, padding=2)
    self.enc_bn1 = nn.BatchNorm2d(128)
    self.enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.enc_bn2 = nn.BatchNorm2d(256)
    self.enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.enc_bn3 = nn.BatchNorm2d(512)
    self.enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.enc_bn4 = nn.BatchNorm2d(1024)
    self.enc_fc_mu = nn.Linear(4 * 4 * 1024, self.hidden_size)
    self.enc_fc_logvar = nn.Linear(4 * 4 * 1024, self.hidden_size)
    self.enc_dropout1 = nn.Dropout(p=.7)

    # Cond encoder layers
    self.cond_enc_conv1 = nn.Conv2d(1, 128, 5, stride=2, padding=2)
    self.cond_enc_bn1 = nn.BatchNorm2d(128)
    self.cond_enc_conv2 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
    self.cond_enc_bn2 = nn.BatchNorm2d(256)
    self.cond_enc_conv3 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
    self.cond_enc_bn3 = nn.BatchNorm2d(512)
    self.cond_enc_conv4 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
    self.cond_enc_bn4 = nn.BatchNorm2d(1024)

    # Decoder layers
    self.dec_upsamp1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    self.dec_conv1 = nn.Conv2d(1024 + self.hidden_size, 512, 3, stride=1, padding=1)
    self.dec_bn1 = nn.BatchNorm2d(512)
    self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv2 = nn.Conv2d(512 * 2, 256, 5, stride=1, padding=2)
    self.dec_bn2 = nn.BatchNorm2d(256)
    self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv3 = nn.Conv2d(256 * 2, 128, 5, stride=1, padding=2)
    self.dec_bn3 = nn.BatchNorm2d(128)
    self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    self.dec_conv4 = nn.Conv2d(128 * 2, 64, 5, stride=1, padding=2)
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

  # define forward pass
  def forward(self, color, greylevel):
    sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
    mu, logvar = self.encoder(color)
    stddev = torch.sqrt(torch.exp(logvar))
    sample = torch.randn(stddev.shape, device=stddev.device)
    z = torch.add(mu, torch.mul(sample, stddev))
    color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
    return mu, logvar, color_out


class MDNMod(nn.Module):
    # define layers
    def __init__(self):
        super(MDNMod, self).__init__()
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

class VAEMod(nn.Module):
    # define layers
    def __init__(self):
        super(VAEMod, self).__init__()
        self.hidden_size = 64
        self.nf = 64
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.enc_conv1 = torch.nn.Conv2d(2, self.nf, 3, stride=2, padding=1)
        self.enc_bn1 = torch.nn.BatchNorm2d(self.nf)
        self.enc_conv2 = torch.nn.Conv2d(self.nf, self.nf * 2, 3, stride=2, padding=1)
        self.enc_bn2 = torch.nn.BatchNorm2d(self.nf * 2)
        self.enc_conv3 = torch.nn.Conv2d(self.nf * 2, self.nf * 4, 3, stride=2, padding=1)
        self.enc_bn3 = torch.nn.BatchNorm2d(self.nf * 4)
        self.enc_conv4 = torch.nn.Conv2d(self.nf * 4, self.nf * 8, 3, stride=2, padding=1)
        self.enc_bn4 = torch.nn.BatchNorm2d(self.nf * 8)
        self.enc_conv5 = torch.nn.Conv2d(self.nf * 8, self.nf * 16, 3, stride=2, padding=1)
        self.enc_bn5 = torch.nn.BatchNorm2d(self.nf * 16)
        self.enc_mu = torch.nn.Conv2d(self.nf * 16, self.hidden_size, 3, stride=2, padding=1)
        self.enc_logvar = torch.nn.Conv2d(self.nf * 16, self.hidden_size, 3, stride=2, padding=1)

        self.upsample1 = Upsample(2)
        self.dec_conv1 = torch.nn.Conv2d(self.hidden_size, self.nf * 16, kernel_size=3, stride=1, padding=1)
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
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def decode(self, x):
        x = self.relu(self.dec_bn1(self.dec_conv1(self.upsample1(x))))
        x = self.relu(self.dec_bn2(self.dec_conv2(self.upsample2(x))))
        x = self.relu(self.dec_bn3(self.dec_conv3(self.upsample3(x))))
        x = self.relu(self.dec_bn4(self.dec_conv4(self.upsample4(x))))
        x = self.relu(self.dec_bn5(self.dec_conv5(self.upsample5(x))))
        x = self.dec_conv6(self.upsample6(x))
        return x

    def forward(self, color):
        mu, logvar = self.encode(color)
        stddev = torch.sqrt(torch.exp(logvar))
        sample = torch.randn(stddev.shape, device=stddev.device)
        z = torch.add(mu, torch.mul(sample, stddev))
        color_out = self.decode(z)
        return mu, logvar, color_out


if __name__ == "__main__":
  from utils import *
  vae = VAEMod()
  mdn = MDNMod()
  print("vae: {} params".format(count_parameters(vae)))
  print("mdn: {} params".format(count_parameters(mdn)))
  print("total {}".format(count_parameters(vae) + count_parameters(mdn)))
