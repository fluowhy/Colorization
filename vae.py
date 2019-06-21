import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

"""
author: 
"""


class VAE(nn.Module):
  
  #define layers
  def __init__(self, nf, device):
    super(VAE, self).__init__()
    self.device = device
    self.hidden_size = 64
    self.enc_end_size = 2
    self.nf = nf # 128

    #Encoder layers
    self.enc_conv1 = nn.Conv2d(2, self.nf, 5, stride=2, padding=2)
    self.enc_bn1 = nn.BatchNorm2d(self.nf)
    self.enc_conv2 = nn.Conv2d(self.nf, self.nf*2, 5, stride=2, padding=2)
    self.enc_bn2 = nn.BatchNorm2d(self.nf*2)
    self.enc_conv3 = nn.Conv2d(self.nf*2, self.nf*4, 5, stride=2, padding=2)
    self.enc_bn3 = nn.BatchNorm2d(self.nf*4)
    self.enc_conv4 = nn.Conv2d(self.nf*4, self.nf*8, 3, stride=2, padding=1)
    self.enc_bn4 = nn.BatchNorm2d(self.nf*8)
    self.enc_fc1 = nn.Linear(self.enc_end_size*self.enc_end_size*self.nf*8, self.hidden_size*2)
    self.enc_dropout1 = nn.Dropout(p=.7)

    #Cond encoder layers
    self.cond_enc_conv1 = nn.Conv2d(1, self.nf, 5, stride=2, padding=2)
    self.cond_enc_bn1 = nn.BatchNorm2d(self.nf)
    self.cond_enc_conv2 = nn.Conv2d(self.nf, self.nf*2, 5, stride=2, padding=2)
    self.cond_enc_bn2 = nn.BatchNorm2d(self.nf*2)
    self.cond_enc_conv3 = nn.Conv2d(self.nf*2, self.nf*4, 5, stride=2, padding=2)
    self.cond_enc_bn3 = nn.BatchNorm2d(self.nf*4)
    self.cond_enc_conv4 = nn.Conv2d(self.nf*4, self.nf*8, 3, stride=2, padding=1)
    self.cond_enc_bn4 = nn.BatchNorm2d(self.nf*8)

    #Decoder layers
    self.dec_upsamp1 = nn.Upsample(scale_factor=2, mode='bilinear')  # scale_factor=4
    self.dec_conv1 = nn.Conv2d(self.nf*8+self.hidden_size, self.nf*4, 3, stride=1, padding=1)
    self.dec_bn1 = nn.BatchNorm2d(self.nf*4)
    self.dec_upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv2 = nn.Conv2d(self.nf*4*2, self.nf*2, 5, stride=1, padding=2)
    self.dec_bn2 = nn.BatchNorm2d(self.nf*2)
    self.dec_upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv3 = nn.Conv2d(self.nf*2*2, self.nf, 5, stride=1, padding=2)
    self.dec_bn3 = nn.BatchNorm2d(self.nf)
    self.dec_upsamp4 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv4 = nn.Conv2d(self.nf*2, int(self.nf/2), 5, stride=1, padding=2)
    self.dec_bn4 = nn.BatchNorm2d(int(self.nf/2))
    self.dec_upsamp5 = nn.Upsample(scale_factor=2, mode='bilinear')
    self.dec_conv5 = nn.Conv2d(int(self.nf/2), 2, 5, stride=1, padding=2)

  def encoder(self, x):
    x = F.relu(self.enc_conv1(x))
    x = self.enc_bn1(x)
    x = F.relu(self.enc_conv2(x))
    x = self.enc_bn2(x)
    x = F.relu(self.enc_conv3(x))
    x = self.enc_bn3(x)
    x = F.relu(self.enc_conv4(x))
    x = self.enc_bn4(x)
    x = x.view(-1, self.enc_end_size*self.enc_end_size*self.nf*8)
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
    x = torch.tanh(self.dec_conv5(x))
    return x
      
  #define forward pass
  def forward(self, color, greylevel):
    """
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
    """
    sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
    mu, logvar = self.encoder(color)
    stddev = torch.sqrt(torch.exp(logvar))
    eps = torch.randn(stddev.size()).normal_().to(self.device)
    z = torch.add(mu, torch.mul(eps, stddev))
    color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
    return mu, logvar, color_out


class VAE96(nn.Module):
  # define layers
  def __init__(self, in_ab, in_l, nf, ld, ks, do=0.7):
    super(VAE96, self).__init__()
    self.ld = ld
    self.relu = torch.nn.ReLU()
    self.tanh = torch.nn.Tanh()

    self.encoder_ab = torch.nn.Sequential(
      torch.nn.Conv2d(in_ab, nf, ks, stride=2, padding=17, dilation=1),  # 64
      torch.nn.BatchNorm2d(nf),
      torch.nn.ReLU(),
      torch.nn.Conv2d(nf, 2*nf, ks, stride=2, padding=1, dilation=1),  # 32
      torch.nn.BatchNorm2d(2*nf),
      torch.nn.ReLU(),
      torch.nn.Conv2d(2*nf, 4*nf, ks, stride=2, padding=1, dilation=1),  # 16
      torch.nn.BatchNorm2d(4*nf),
      torch.nn.ReLU(),
      torch.nn.Conv2d(4*nf, 8*nf, ks, stride=2, padding=1, dilation=1),  # 8
      torch.nn.BatchNorm2d(8*nf),
      torch.nn.ReLU(),
      torch.nn.Conv2d(8*nf, 16*nf, ks, stride=2, padding=1, dilation=1),  # 4
      torch.nn.BatchNorm2d(16*nf),
      torch.nn.ReLU(),
      torch.nn.Conv2d(16*nf, 32*nf, ks, stride=2, padding=1, dilation=1),  # 2
      torch.nn.BatchNorm2d(32*nf),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32 * nf, 64 * nf, ks, stride=2, padding=1, dilation=1),  # 1
      torch.nn.BatchNorm2d(64 * nf),
      torch.nn.ReLU(),
      torch.nn.Dropout(do)
    )

    self.fc_ab = torch.nn.Linear(64 * nf, ld * 2)
    
    self.cond_enc_conv1 = torch.nn.Conv2d(in_l, nf, ks, stride=2, padding=17, dilation=1)
    self.cond_enc_bn1 = torch.nn.BatchNorm2d(nf)
    self.cond_enc_conv2 = torch.nn.Conv2d(nf, 2*nf, ks, stride=2, padding=1, dilation=1)
    self.cond_enc_bn2 = torch.nn.BatchNorm2d(2*nf)
    self.cond_enc_conv3 = torch.nn.Conv2d(2*nf, 4*nf, ks, stride=2, padding=1, dilation=1)
    self.cond_enc_bn3 = torch.nn.BatchNorm2d(4*nf)
    self.cond_enc_conv4 = torch.nn.Conv2d(4*nf, 8*nf, ks, stride=2, padding=1, dilation=1)
    self.cond_enc_bn4 = torch.nn.BatchNorm2d(8*nf)
    self.cond_enc_conv5 = torch.nn.Conv2d(8*nf, 16*nf, ks, stride=2, padding=1, dilation=1)
    self.cond_enc_bn5 = torch.nn.BatchNorm2d(16*nf)
    self.cond_enc_conv6 = torch.nn.Conv2d(16 * nf, 32 * nf, ks, stride=2, padding=1, dilation=1)
    self.cond_enc_bn6 = torch.nn.BatchNorm2d(32 * nf)
    self.cond_enc_conv7 = torch.nn.Conv2d(32 * nf, 64 * nf, ks, stride=2, padding=1, dilation=1)
    self.cond_enc_bn7 = torch.nn.BatchNorm2d(64 * nf)

    self.dec_conv1 = torch.nn.ConvTranspose2d(ld, 64 * nf, 2, stride=1, padding=0, dilation=1)  # 2
    self.dec_bn1 = torch.nn.BatchNorm2d(64*nf)
    self.dec_conv2 = torch.nn.ConvTranspose2d(64 * nf, 16 * nf, 2, stride=2, padding=0, dilation=1)  # 4
    self.dec_bn2 = torch.nn.BatchNorm2d(16 * nf)
    self.dec_conv3 = torch.nn.ConvTranspose2d(32 * nf, 8 * nf, 2, stride=2, padding=0, dilation=1)  # 8
    self.dec_bn3 = torch.nn.BatchNorm2d(8 * nf)
    self.dec_conv4 = torch.nn.ConvTranspose2d(16 * nf, 4 * nf, 2, stride=2, padding=0, dilation=1)  # 16
    self.dec_bn4 = torch.nn.BatchNorm2d(4 * nf)
    self.dec_conv5 = torch.nn.ConvTranspose2d(8 * nf, 2 * nf, 2, stride=2, padding=0, dilation=1)  # 32
    self.dec_bn5 = torch.nn.BatchNorm2d(2 * nf)
    self.dec_conv6 = torch.nn.ConvTranspose2d(4 * nf, 2 * nf, 2, stride=2, padding=0, dilation=1)  # 64
    self.dec_bn6 = torch.nn.BatchNorm2d(2 * nf)
    self.dec_conv7 = torch.nn.ConvTranspose2d(2 * nf, in_ab, 2, stride=2, padding=16, dilation=1)  # 96
    self.dec_bn7 = torch.nn.BatchNorm2d(in_ab)

  def encode(self, x):
    h = self.encoder_ab(x).squeeze()
    h = self.fc_ab(h)
    mu = h[..., :self.ld]
    logvar = h[:, self.ld:]
    return mu, logvar

  def cond_encoder(self, x):
    x = self.relu(self.cond_enc_bn1(self.cond_enc_conv1(x)))
    sc_feat32 = self.cond_enc_bn2(self.cond_enc_conv2(x))
    x = self.relu(sc_feat32)
    sc_feat16 = self.cond_enc_bn3(self.cond_enc_conv3(x))
    x = self.relu(sc_feat16)
    sc_feat8 = self.cond_enc_bn4(self.cond_enc_conv4(x))
    x = self.relu(sc_feat8)
    sc_feat4 = self.cond_enc_bn5(self.cond_enc_conv5(x))
    x = self.relu(sc_feat4)
    x = self.relu(self.cond_enc_bn6(self.cond_enc_conv6(x)))
    x = self.relu(self.cond_enc_bn7(self.cond_enc_conv7(x)))
    return x, sc_feat32, sc_feat16, sc_feat8, sc_feat4

  def decoder(self, z, sc_feat32, sc_feat16, sc_feat8, sc_feat4):
    z = z.reshape(z.shape[0], self.ld, 1, 1)
    z = self.relu(self.dec_bn1(self.dec_conv1(z)))
    z = self.relu(self.dec_bn2(self.dec_conv2(z)))
    z = torch.cat((z, sc_feat4), 1)
    z = self.relu(self.dec_bn3(self.dec_conv3(z)))
    z = torch.cat((z, sc_feat8), 1)
    z = self.relu(self.dec_bn4(self.dec_conv4(z)))
    z = torch.cat((z, sc_feat16), 1)
    z = self.relu(self.dec_bn5(self.dec_conv5(z)))
    z = torch.cat((z, sc_feat32), 1)
    z = self.relu(self.dec_bn6(self.dec_conv6(z)))
    z = self.tanh(self.dec_bn7(self.dec_conv7(z)))
    return z

  # define forward pass
  def forward(self, color, greylevel):
    """
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
    """
    z_cond, sc_feat32, sc_feat16, sc_feat8, sc_feat4 = self.cond_encoder(greylevel)
    mu, logvar = self.encode(color)
    stddev = torch.sqrt(torch.exp(logvar))
    sample = torch.randn(stddev.shape, device=stddev.device)
    z = torch.add(mu, torch.mul(sample, stddev))
    color_out = self.decoder(z, sc_feat32, sc_feat16, sc_feat8, sc_feat4)
    return mu, logvar, color_out, z_cond


