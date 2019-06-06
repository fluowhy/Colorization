import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from model import *
from utils import *

device = "cuda:0"
nin = 4
ld = 2
k = 4
lr = 2e-4
wd = 0.
epochs = 200
bs = 100

N = 10000

u0 = np.ones(nin)
s = np.ones(nin)

x0 = np.random.multivariate_normal(mean=u0, cov=np.diag(s), size=N)
x1 = np.random.multivariate_normal(mean=- u0, cov=np.diag(s), size=N)
x = np.vstack((x0, x1))

z0 = np.random.normal(1, 1, size=(N, ld))
z1 = np.random.normal(- 1, 1, size=(N, ld))
z = np.vstack((z0, z1))

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



train_dataset = torch.utils.data.TensorDataset(x_train, z_train)
test_dataset = torch.utils.data.TensorDataset(x_test, z_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=True)

gmm = GMM(nin=nin, nh=10, k=k, ld=ld).to(device)
optimizer = torch.optim.Adam(gmm.parameters(), lr=lr, weight_decay=wd)
gmml = GMMLoss(ld)

best_loss = np.inf
for epoch in range(epochs):
	train_loss = train_gmm(gmm, optimizer, train_loader, gmml)
	test_loss = eval_gmm(gmm, test_loader, gmml)
	best_loss = save_or_not(test_loss, best_loss, gmm, "gmm")
	print_epoch(epoch, train_loss, test_loss)

gmm.load_state_dict(torch.load("models/gmm.pth"))
gmm.eval()
with torch.no_grad():
	mu, _, _ = gmm(x_test)
mu = mu.cpu().numpy()

plt.figure()
for i in range(k):
	plt.scatter(mu[:, 0, i], mu[:, 1, i], marker=".")
plt.show()