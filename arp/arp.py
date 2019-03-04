import numpy as np
from collections.abc import Iterable


class ARProcess:
    def __init__(self, p=0, alpha=0, size=1, seed=None):
        """
        Args:
            p: an integer process order
            alpha: a scalar or a vector of smoothing parameters alpha_k
            m: dimensionality of noise vector, e.g. dimensionality of action space
        """
        if isinstance(alpha, Iterable):
            assert len(alpha) == p, "Length of alpha vector must be equal to the process order"
        else:
            alpha = [alpha] * p
        for val in alpha:
            assert 0 <= val < 1, "alpha values must lie within [0, 1) interval"
        self.alpha = alpha
        self.p = p
        self.size = size
        self.phi = self.compute_phi(self.alpha)
        self.acv, self.sigma_z = self.solve_yule_walker(self.phi)
        self.reset(seed)

    def compute_phi(self, alpha):
        def polynomial_coeffs(alpha):
            if len(alpha) == 1:
                return [-alpha[0]]
            coeffs = polynomial_coeffs(alpha[:-1])
            coeffs.append(coeffs[-1] - alpha[-1])
            for j in range(len(coeffs) - 2, 0, -1):
                coeffs[j] = coeffs[j - 1] - alpha[-1] * coeffs[j]
            coeffs[0] *= -alpha[-1]
            return coeffs
        phi = polynomial_coeffs(alpha)
        phi.reverse()
        return -np.expand_dims(np.array(phi), axis=1)

    def solve_yule_walker(self, phi):
        p = len(phi)
        A = np.zeros((p, p))
        for r in range(p):
            for j in range(r):
                A[r, j] -= phi[r - j - 1]
            for j in range(r + 1, p):
                A[r, j - r - 1] -= phi[j]
            A[r, r] += 1
        acv = np.linalg.solve(A, phi)    # autocovariance function
        sigma_z = np.sqrt(1 - np.sum(phi * acv))
        return acv, sigma_z

    def acf(self, k):
        """Computes k values of autocorrelation function ro(tau), tau = 1, .., k.

        Equivalent to autocovariance function since var(X_t) = 1
        """
        acf = list(self.acv.copy())
        for j in range(len(acf), k):
            new_val = 0
            for i in range(len(self.phi)):
                new_val += self.phi[i] * acf[abs(i - j + 1)]
            acf.append(new_val)
        return np.array(acf)

    def reset(self, seed=None):
        self.history = np.zeros((self.p, self.size))
        if not seed is None:
            np.random.seed(seed)

    def step(self):
        rnd = np.random.normal(size=self.size)
        h = np.sum(self.history[::-1] * self.phi, axis=0)
        x_t = np.sum(self.history[::-1] * self.phi, axis=0) + rnd * self.sigma_z
        self.history = np.vstack([self.history, x_t])[1:]
        return x_t, h

if __name__ == "__main__":
    ar = ARProcess(3, 0.5, 2)
    for i in range(100):
        print(ar.step())