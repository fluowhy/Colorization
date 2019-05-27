import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from model import *
from utils import *

device = "cpu"
nin = 4
ld = 2
k = 3
lr = 2e-4
wd = 0.
epochs = 100
bs = 10

N = 1000

u0 = np.ones(nin)
s = np.ones(nin)

x0 = np.random.multivariate_normal(mean=u0, cov=np.diag(s), size=N)
x1 = np.random.multivariate_normal(mean=- u0, cov=np.diag(s), size=N)

x = np.vstack((x0, x1))
z = np.random.normal(size=(N, ld))

idx = np.arange(N)
idx_train, idx_test = train_test_split(idx, test_size=0.2, shuffle=True)

x_train = x[idx_train]
x_test = x[idx_test]
z_train = z[idx_train]
z_test = z[idx_test]

x_train = torch.tensor(x_train, dtype=torch.float, device=device)
x_test = torch.tensor(x_test, dtype=torch.float, device=device)
z_train = torch.tensor(z_train, dtype=torch.float, device=device)
z_test = torch.tensor(z_test, dtype=torch.float, device=device)

def gmmloss(mu, log_s2, w, z):
	r = (- 0.5 * ((z.unsqueeze(-1) - mu).pow(2) / log_s2.exp()).sum(dim=1)).exp()
	det = log_s2.exp().prod(dim=1).sqrt()
	pr = (r / det * w).sum(dim=1)
	return (- torch.log(pr)).mean()

train_dataset = torch.utils.data.TensorDataset(x_train, z_train)
test_dataset = torch.utils.data.TensorDataset(x_test, z_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)

gmm = GMM(nin=nin, nh=10, k=k, ld=ld).to(device)
optimizer = torch.optim.Adam(gmm.parameters(), lr=lr, weight_decay=wd)
gmml = GMMLoss(ld)

for epoch in range(epochs):
    train_loss = train_gmm(gmm, optimizer, train_loader, gmml)
    test_loss = eval_gmm(gmm, test_loader, gmml)
    print_epoch(epoch, train_loss, test_loss)
