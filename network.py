import numpy as np
import torch
import torch.nn as nn
import prox_tv
from utils import ProxTV_l1, _soft_th_tensor, _soft_th_numpy, _loss_fn1, _MSELoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LPGMISTA(nn.Module):
    def __init__(self, n, m, A, max_iter, lmbd_1, lmbd_2, learn_th=1):
        """
            # Arguments
            m: int, dimensions of the measurement
            n: int, dimensions of the signal
            W_x: array
            W_y: array
            max_iter:int, max number of internal iteration
            L: Lipschitz const
            lmbd_1: sparsity penalty
            lmbd_2: gradient sparsity penalty
         """

        super(LPGMISTA, self).__init__()
        self.A = A.to(dtype=torch.float64)
        self.I_k = torch.eye(A.shape[1], dtype=torch.float64).to(device)
        self.W_x = nn.Parameter(torch.zeros((n, n), dtype=torch.float64, device=device), requires_grad=True)
        self.W_y = nn.Parameter(torch.zeros((n, m), dtype=torch.float64, device=device), requires_grad=True)
        if learn_th:
            self.u = nn.Parameter(torch.tensor([1.0], dtype=torch.float64, device=device), requires_grad=True)
            self.t = nn.Parameter(torch.tensor([1.0], dtype=torch.float64, device=device), requires_grad=True)
        else:
            self.u = 1 / torch.norm(self.A, p=2) ** 2
            self.t = self.u
        self.max_iter = max_iter
        self.lmbd_1 = lmbd_1
        self.lmbd_2 = lmbd_2

    def weights_init(self, learn_th=1):
        u_val = 1 / torch.norm(self.A, p=2) ** 2 # # this can be tuned according to the theories of PGM-ISTA
        t_val = .9 * u_val  # this can be tuned according to the theories of PGM-ISTA
        if learn_th:
            self.u.data = torch.tensor([u_val])
            self.t.data = torch.tensor([t_val])
        self.W_x.data = self.I_k - torch.matmul(self.A.T, self.A) * u_val
        self.W_y.data = self.A.T * u_val

    def forward(self, y):
        self.lmbd_1 = torch.as_tensor(self.lmbd_1, dtype=torch.float64)
        self.lmbd_2 = torch.as_tensor(self.lmbd_2, dtype=torch.float64)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64, device=device)
        if self.t.data <= 0:
            self.t = torch.nn.Parameter(torch.tensor(1e-30, dtype=self.t.dtype, device=self.t.device))
        if self.u.data <= 0:
            self.u = torch.nn.Parameter(torch.tensor(1e-30, dtype=self.u.dtype, device=self.u.device))
        x = self.t / self.u * _soft_th_tensor(torch.matmul(y, self.W_y.T), self.lmbd_1 * self.u)
        x = ProxTV_l1.apply(x, self.lmbd_2 * self.t)

        if self.max_iter == 1:
            return x

        for _ in range(self.max_iter - 1):
            if self.t.data <= 0:
                self.t = torch.nn.Parameter(torch.tensor(1e-30, dtype=self.t.dtype, device=self.t.device))
            if self.u.data <= 0:
                self.u = torch.nn.Parameter(torch.tensor(1e-30, dtype=self.u.dtype, device=self.u.device))
            Wx_x = torch.matmul(x, self.W_x.T)
            Wy_y = torch.matmul(y, self.W_y.T)
            u_1 = self.t / self.u * _soft_th_tensor(Wx_x + Wy_y, self.lmbd_1 * self.u)
            x = (1 - self.t / self.u) * x + u_1
            x = ProxTV_l1.apply(x, self.lmbd_2 * self.t)

        return x


def train_lpgmista(Y, A, x0, max_iter=10, lmbd_1=.05, lmbd_2=3):
    m, n = A.shape
    Y = torch.from_numpy(Y)
    Y = Y.to(device)

    x0 = torch.from_numpy(x0)
    x0 = x0.to(device)

    A = torch.from_numpy(A)
    A = A.to(device)

    net = LPGMISTA(n, m, A, max_iter, lmbd_1, lmbd_2)
    net = net.float().to(device)
    net.weights_init()

    # build the optimizer
    learning_rate = 3e-6  # this can be tuned
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    all_epoch = 200  # this can be tuned
    loss_list = []
    for epoch in range(all_epoch):
        optimizer.zero_grad()
        # get the outputs
        X_h = net(Y)
        loss = _MSELoss(X_h, x0)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_list.append(loss.detach().data)
        print(f"current loss:{loss_list[-1]:.2e}, "
              f"current epoch:{epoch + 1}/{all_epoch}, ")

    return net, loss_list


def pgmista(Y, A, max_iter=1000, lmbd_1=.01, lmbd_2=.25):
    x_old = np.zeros((Y.shape[0], A.shape[1]))
    u = 1 / np.linalg.norm(A, ord=2) ** 2
    t = u
    I_k = np.eye(A.shape[1])
    W_x = I_k - A.T.dot(A) * u
    W_y = A.T * u
    recon_errors = []
    for i in range(max_iter):
        x_soft = t / u * _soft_th_numpy(x_old.dot(W_x.T) + Y.dot(W_y.T), lmbd_1 * u)
        x_temp = (1 - t / u) * x_old + x_soft
        x_new = np.zeros(x_old.shape)
        for j in range(x_old.shape[0]):
            x_new[j] = prox_tv.tv1_1d(x_temp[j], lmbd_2 * t)
        x_old = x_new
        loss = _loss_fn1(x_new, A, lmbd_1, lmbd_2, Y)
        recon_errors.append(loss)
    return x_new, recon_errors
