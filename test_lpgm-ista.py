import matplotlib.pyplot as plt
from network import train_lpgmista, pgmista, LPGMISTA
import torch
from utils import check_random_state, save_txt
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from scipy.io import loadmat, savemat

rng = check_random_state(200)
sr = .5
n = 256 # x.shape[1]
m = int(n * sr)
A = rng.randn(m, n)

# load models
load_path = 'model_256_2.pth'
M = torch.from_numpy(A)
M = M.to(device)
m, n = M.shape
net = LPGMISTA(n, m, M, max_iter=2, lmbd_1=.01, lmbd_2=.25)
net.load_state_dict(torch.load(load_path))
net.eval()

# test models
with torch.no_grad():
    data = loadmat('test_data.mat', mat_dtype=True)
    test_data = data['test_data'].T
    test_data_y = test_data.dot(A.T)
    t__ = time.time()
    x0_recon = net(test_data_y)
    myt_ = time.time() - t__
    str_info = "layer length =" + str(2) + "; CPU time=" + str(myt_) + "\n"
    save_txt(str_info)
    x0_recon = x0_recon.detach().cpu().numpy()
    savemat('rec_256_2.mat', {'rec_256_2': x0_recon})

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(test_data.T, label='real')
plt.plot(x0_recon.T, '.-', label='LPGM-ISTA')
plt.legend()

# PGM-ISTA
Z_recon, recon_loss = pgmista(test_data_y, A, max_iter=2, lmbd_1=.01, lmbd_2=.25)
plt.subplot(2, 1, 2)
plt.plot(test_data.T, label='real')
plt.plot(Z_recon.T, '--', label='PGM-ISTA')
plt.legend()
plt.show()
