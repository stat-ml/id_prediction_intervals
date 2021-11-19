import numpy as np
import scipy
import scipy.stats

import torch
from tqdm.auto import tqdm

from .datasets.regression_datamodule import safe_numpy


def flatten_grad(g):
    return torch.cat([p.view(-1) for p in g])


def numel(m: torch.nn.Module, only_trainable: bool = True):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


class ImportantDirectionsPytorch:
    def __init__(self, model, rank, max_rows=None, alpha_final=0.01, tol=1e-6, torch_dtype=torch.float32, device='cpu'):
        self.model = model
        self.rank = rank
        self.m = numel(self.model)
        self.n = None
        self.tol = tol

        self.alpha_0 = alpha_final
        self.alpha_final = 0

        self.device = device
        self.torch_dtype = torch_dtype

        self.max_rows = 2 * self.rank if max_rows is None else max_rows

        self.B = torch.zeros((self.max_rows, self.m), dtype=self.torch_dtype, device=self.device)

        # Pre-allocate for SVD
        self.k = min(self.m, self.max_rows)
        self.U = torch.zeros((self.max_rows, self.k), dtype=self.torch_dtype, device=self.device)
        self.D = torch.zeros(self.k, dtype=self.torch_dtype, device=self.device)
        self.Vt = torch.zeros((self.k, self.m), dtype=self.torch_dtype, device=self.device)

        # Final approximation
        self.B_final = self.B[:self.rank, :]

        self.k_final = min(self.m, self.max_rows, self.rank)
        self.U_final = self.U[:self.rank, :self.k_final]
        self.D_final = self.D[:self.k_final]
        self.DV = torch.zeros(self.k_final, dtype=self.torch_dtype, device=self.device)
        self.Vt_final = self.Vt[:self.k_final, :self.m]

        self.first_zero_row = 0

        # For intervals
        self.residuals2 = None
        self.s_hat = None
        self.p_star = None
        self.tq = None

    def fit(self, X, y):
        self.n = X.shape[0]
        y_pred = torch.zeros_like(y)
        self.model.to(self.device)
        self.init_approximation()
        pbar = tqdm(range(self.n), desc="ImpDirs.fit")
        for i in pbar:
            xb, yb = X[i], y[i]

            yb_pred = self.model(xb)
            gi = flatten_grad(torch.autograd.grad(yb_pred, self.model.parameters()))
            y_pred[i] = yb_pred.detach()

            self.update_approximation(gi, pbar)
        self.model.to('cpu')

        self.finilize_approximation()

        self.residuals2 = safe_numpy((y_pred - y) ** 2)

        # Use final approximation
        with torch.no_grad():
            torch.linalg.svd(self.B_final, full_matrices=False, out=(self.U_final, self.D_final, self.Vt_final))

        self.compute_estimates()

        print(f"{self.p_star=:.2f}, {self.s_hat=:.2f}, {self.tq=:.2f}, {self.alpha_0=}, {self.alpha_final=}")

    def init_approximation(self):
        self.first_zero_row = 0

    def finilize_approximation(self):
        if self.first_zero_row > self.rank:
            self.update_approximation_inner(None)

    def compute_estimates(self):
        D2 = self.D_final ** 2 + self.alpha_final
        D2a = self.D_final ** 2 + self.alpha_0 + self.alpha_final
        self.DV = D2 / (D2a ** 2)
        Dh = D2 / D2a
        Dh2 = D2 ** 2 / (D2a ** 2)
        self.p_star = safe_numpy((2 * Dh - Dh2).sum()).item()
        self.s_hat = np.sqrt(self.residuals2.sum() / (self.n - self.p_star))
        self.tq = scipy.stats.t.ppf(0.975, self.n - self.p_star)

    def save_to_file(self, s):
        U_final, D_final, Vt_final, DV = map(safe_numpy, (self.U_final, self.D_final, self.Vt_final, self.DV))
        np.savez(s, U_final=U_final, D_final=D_final, Vt_final=Vt_final, DV=DV,
                 residuals2=self.residuals2, alpha_0=self.alpha_0, alpha_final=self.alpha_final,
                 p_star=self.p_star, s_hat=self.s_hat, tq=self.tq)

    def update_approximation(self, g, pbar):
        if self.first_zero_row >= self.max_rows:
            self.update_approximation_inner(pbar)
        self.B[self.first_zero_row] = g
        self.first_zero_row += 1

    def update_approximation_inner(self, pbar):
        # Update approximation
        with torch.no_grad():
            torch.linalg.svd(self.B, full_matrices=False, out=(self.U, self.D, self.Vt))
        s = safe_numpy(self.D)
        s = s[s > self.tol]
        N = len(s)
        # pbar.set_description(f"Smallest sv: {s[-1]:10.4f}", refresh=True)
        # pbar.set_postfix(f"min={s.min():6f} | max={s.max():6f} | r={N:5d}", refresh=True)
        if pbar is not None:
            pbar.set_postfix({"min": s.min(), "max": s.max(), "r": N})
        if N < self.rank:
            # Actual rank less than required, using all found singular values
            idx_selected = np.arange(N)
            s_selected = s
        else:
            # More singular values than required, need to cut off and shrink
            #score = (s ** 2 + self.alpha_final) / (s ** 2 + self.alpha_0 + self.alpha_final) ** 2
            score = (s ** 2) / (s ** 2 + self.alpha_0) ** 2
            idx_sorted = score.argsort()[::-1]
            idx_top = idx_sorted[:self.rank]
            idx_max = np.argmax(score)

            # diag_H = s ** 2 / (s ** 2 + self.alpha_final)
            # score_2 = 2. * diag_H - diag_H ** 2
            # idx_sorted_2 = score_2.argsort()[::-1]
            # idx_top_2 = idx_sorted_2[:self.rank]

            # idx_comb = np.array(list(set(idx_top) | set(idx_top_2)))
            # pbar.write(f"{idx_comb=}")
            # pbar.write(f"{idx_top[:5]=}, {idx_top_2[:5]=}")

            s_selected = s[idx_top]
            s_smallest = s_selected.min()
            s_shrunk_2 = s_selected ** 2 - s_smallest ** 2
            assert np.all(s_shrunk_2 >= 0)
            s_shrunk = np.sqrt(s_shrunk_2)

            self.alpha_final += s_smallest ** 2 / 2.

            s_selected = s_shrunk.copy()
            idx_selected = idx_top.copy()
        self.first_zero_row = len(idx_selected)
        s_selected_t = torch.tensor(s_selected, dtype=self.torch_dtype, device=self.device)
        torch.matmul(torch.diag(s_selected_t),
                     self.Vt[idx_selected, :],
                     out=self.B[:self.first_zero_row, :])
        self.B[self.first_zero_row:, :] = 0

    def predict_to_numpy(self, X, y):
        n = X.shape[0]
        y_pred = torch.zeros_like(y)
        y_l = torch.zeros_like(y)
        y_u = torch.zeros_like(y)
        #y_std = torch.zeros_like(y)
        self.model.to(self.device)
        pbar = tqdm(range(n), desc="ImpDirs.predict")
        for i in pbar:
            xb, yb = X[i], y[i]
            yb_pred = self.model(xb)
            y_pred[i] = yb_pred.detach()
            gi = flatten_grad(torch.autograd.grad(yb_pred, self.model.parameters()))

            #print(f"{gi.shape=}, {self.Vt.shape=}, {self.DV.shape=}")

            GV = torch.mv(self.Vt_final, gi)
            prod = (self.DV * ((GV) ** 2)).sum()
            sqrt_term = torch.sqrt(1 + prod)
            width = self.tq * self.s_hat * sqrt_term
            y_l[i] = yb_pred - width
            y_u[i] = yb_pred + width
        self.model.to('cpu')

        return safe_numpy(y_pred), safe_numpy(y_l), safe_numpy(y_u)


class IntervalsExactPytorch(ImportantDirectionsPytorch):
    def __init__(self, model, M, alpha_final=0.01, tol=1e-6, torch_dtype=torch.float32, device='cpu'):
        super().__init__(model, rank=M, max_rows=M, alpha_final=alpha_final, tol=tol,
                         torch_dtype=torch_dtype, device=device)

    def init_approximation(self):
        pass

    def update_approximation(self, g, pbar):
        self.B += torch.outer(g, g)

    def finilize_approximation(self):
        pass

    def update_approximation_inner(self, pbar):
        pass
