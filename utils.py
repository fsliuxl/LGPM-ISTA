import torch
from torch.nn.functional import relu as relu_tensor
import prox_tv
import numbers
import numpy as np

def _MSELoss(x, y, temp=1):
    n_samples = x.shape[0]
    residual = x - y
    if temp:
        loss = (residual * residual).sum()
    else:
        loss = torch.abs(residual).sum()
    return loss / n_samples


def _loss_fn1(x, A, lbda_1, lbda_2, y):
    """
    Loss function for the primal.
        :math:`L(x) = 1/2 ||y - Ax||_2^2 + lbda_1 ||x||_1 + lbda_2 ||Dx||_1`
    """
    n_samples = x.shape[0]
    residual = np.dot(x, A.T) - y
    loss = 0.5 * (residual * residual).sum()
    loss = loss + lbda_1 * abs(y).sum() + lbda_2 * abs(y[:, 1:] - y[:, :-1]).sum()
    return loss / n_samples


def _soft_th_tensor(z, mu):
    y = torch.sign(z) * relu_tensor(z.abs() - mu)
    if isinstance(y, torch.Tensor):
        y = y.double()
    else:
        y = torch.tensor(y, dtype=torch.float64)
    return y

class ProxTV_l1(torch.autograd.Function):
    """
    Custom autograd Function wrapper for the prox_tv.
    """

    @staticmethod
    def forward(ctx, x, lbda):
        # Convert input to numpy array to use the prox_tv library
        device = x.device
        x = x.detach().cpu().data

        # The regularization can be learnable or a float
        if isinstance(lbda, torch.Tensor):
            lbda = lbda.detach().cpu().data

        # Get back a tensor for the output and save it for the backward pass
        output = check_tensor(
            np.array([prox_tv.tv1_1d(xx, lbda) for xx in x]),
            device=device, requires_grad=True,
        )
        z = output - torch.functional.F.pad(output, (1, 0))[..., :-1]
        ctx.save_for_backward(torch.sign(z))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Compute the gradient of proxTV using implicit gradient."""
        batch_size, n_dim = grad_output.shape
        sign_z, = ctx.saved_tensors
        device = grad_output.device
        S = sign_z != 0
        S[:, 0] = True
        sign_z[:, 0] = 0
        # XXX do clever computations
        L = torch.triu(torch.ones((n_dim, n_dim), dtype=torch.float64,
                                  device=device))

        grad_x, grad_lbda = [], []
        for i in range(batch_size):
            L_S = L[:, S[i]]  # n_dim x |S|
            grad_u = grad_output[i].matmul(L_S)  # 1 x |S|
            H_S = torch.inverse(L_S.t().matmul(L_S))
            grad_x.append(grad_u.matmul(H_S.matmul(L_S.t())))
            grad_lbda.append(grad_u.matmul(H_S.matmul(-sign_z[i][S[i]])))
        grad_x = torch.stack(grad_x)
        grad_lbda = torch.stack(grad_lbda)
        return (grad_x, grad_lbda)


def check_tensor(*arrays, device=None, dtype=torch.float64,
                 requires_grad=None):
    n_arrays = len(arrays)
    result = []
    for x in arrays:
        initial_type = type(x)
        if isinstance(x, np.ndarray) or isinstance(x, numbers.Number):
            x = torch.tensor(x)
        assert isinstance(x, torch.Tensor), (
            f"Invalid type {initial_type} in check_tensor. "
            "Should be in {'ndarray, int, float, Tensor'}."
        )
        x = x.to(device=device, dtype=dtype)
        if requires_grad is not None:
            x.requires_grad_(requires_grad)
        result.append(x)

    return tuple(result) if n_arrays > 1 else result[0]


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int, random-instance, (default=None), random-instance
        or random-seed used to initialize the random-instance

    Return
    ------
    random_instance : random-instance used to initialize the analysis
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(f'{seed} cannot be used to seed a '  # noqa: E999
                     f'numpy.random.RandomState instance')


def _soft_th_numpy(z, mu):
    return np.sign(z) * np.maximum(np.abs(z) - mu, 0.0)

def save_txt(str_info):
    path= "log_time.txt"
    # os.makedirs(os.path.dirname(path),exist_ok=True)
    with open(path,"a") as f:
        f.write(str_info)