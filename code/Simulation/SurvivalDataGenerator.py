import numpy as np
from mv_laplace import MvLaplaceSampler
from scipy.stats import logistic

class DataGenerator:
    def __init__(self, *, N=1000, strata_size=10, beta=1, MaskRate=0.5, verbose=False, Missing_lambda=None):
        self.N = N
        self.beta = beta
        self.strata_size = strata_size
        self.totalStrataNumber = int(N / strata_size)
        self.MaskRate = MaskRate
        self.verbose = verbose
        self.Missing_lambda = Missing_lambda

    def GenerateX(self):
        mean = [1/2, -1/3]
        cov = [[1, 1/2], [1/2, 1]]
        X1_2 = np.random.multivariate_normal(mean, cov, self.N)

        loc = [0, 1/np.sqrt(3)]
        cov = [[1, 1/np.sqrt(2)], [1/np.sqrt(2), 1]]
        sampler = MvLaplaceSampler(loc, cov)
        X3_4 = sampler.sample(self.N)

        p = 1/3
        X5 = np.random.binomial(1, p, self.N)
        X = np.hstack((X1_2, X3_4, X5.reshape(-1, 1)))
        return X

    def GenerateU(self):
        return np.random.normal(0, np.sqrt(0.2), self.N).reshape(-1, 1)

    def GenerateS(self):
        S = np.zeros(self.N)
        for i in range(self.totalStrataNumber):
            S[self.strata_size * i:self.strata_size * (i + 1)] = i + 1
        return S.reshape(-1, 1)

    def GenerateZ(self):
        Z = []
        half_strata_size = self.strata_size // 2
        for i in range(self.totalStrataNumber):
            strata = np.array([0.0] * half_strata_size + [1.0] * half_strata_size)
            np.random.shuffle(strata)
            Z.append(strata)
        return np.concatenate(Z).reshape(-1, 1)

    def GenerateEps(self):
        return np.random.normal(0, np.sqrt(0.2), self.N)

    def GenerateStrataEps(self):
        eps = []
        for i in range(self.totalStrataNumber):
            eps.append(np.full(self.strata_size, np.random.normal(0, np.sqrt(0.1))))
        return np.concatenate(eps).reshape(-1,)

    def GenerateTInter(self, T):
        biases = []
        for i in range(self.totalStrataNumber):
            strata = T[i * self.strata_size: (i + 1) * self.strata_size, 0]
            biases.append(np.full(self.strata_size, (10 / 3) * np.mean(strata)))
        return np.concatenate(biases).reshape(-1,)

    def GenerateCInter(self, X):
        biases = []
        for i in range(self.totalStrataNumber):
            strata = X[i * self.strata_size: (i + 1) * self.strata_size, 0]
            biases.append(np.full(self.strata_size, (10 / 3) * np.mean(strata)))
        return np.concatenate(biases).reshape(-1,)

    def GenerateXInter(self, X):
        biases = []
        for i in range(self.totalStrataNumber):
            strata = X[i * self.strata_size: (i + 1) * self.strata_size, 0]
            biases.append(np.full(self.strata_size, (10 / 3) * np.mean(strata)))
        return np.concatenate(biases).reshape(-1,)

    def GenerateT(self, X, U, Z, StrataEps, Eps):
        sum1 = np.sum(X, axis=1) / np.sqrt(5)
        X_rep = X[:, None, :]
        sum2 = np.sum(X_rep * logistic.cdf(1 - X_rep), axis=(1, 2)) / np.sqrt(25)
        sum4 = np.sum(np.abs(X), axis=1) / np.sqrt(5)

        lambda_T = self.beta * Z.flatten() * (1 + X[:, 0] + sum4) + sum1 + sum2 + U.flatten() + StrataEps + Eps
        lambda_T = 8 * np.abs(lambda_T)  # Apply exponential to ensure positivity
        print("Mean of lambda_T: ", np.mean(lambda_T))        
        T = np.random.poisson(lambda_T).reshape(-1, 1)
        return T

    def GenerateC(self, X, U, T):
        T = T.flatten()[:self.N]
        sum5 = np.sum(np.arange(1, 6) * X, axis=1) / np.sqrt(5)
        sum6 = np.sum(np.arange(1, 6) * np.cos(X), axis=1) / np.sqrt(5)
        XInter = self.GenerateXInter(X)  # Incorporate XInter

        lambda_C = sum5 + sum6 + 10 * logistic.cdf(T) + U.flatten() + XInter
        lambda_C = 0.6 * np.abs(lambda_C)  # Apply exponential to ensure positivity
        C = np.random.poisson(lambda_C)
        C = np.where(C < 10, C, 10)
        print("Mean of lambda_C: ", np.mean(lambda_C))
        return C.reshape(-1, 1)

    def GenerateM(self, X, U, T, C):
        sum5 = np.sum(np.arange(1, 6) * np.cos(X), axis=1) / np.sqrt(5)
        TInter = self.GenerateTInter(T)
        CInter = self.GenerateCInter(C)
        XInter = self.GenerateXInter(X)
        lambda_M = sum5 + 10 * logistic.cdf(T.flatten()) + 10 * logistic.cdf(C.flatten()) + U.flatten() + TInter + CInter + XInter
        threshold = np.percentile(lambda_M, 100 * (1 - self.MaskRate))
        M = (lambda_M > threshold).astype(int)
        return M.reshape(-1, 1)

    def GenerateData(self):
        X = self.GenerateX()
        Z = self.GenerateZ()
        U = self.GenerateU()
        S = self.GenerateS()
        Eps = self.GenerateEps()
        StrataEps = self.GenerateStrataEps()
        T = self.GenerateT(X, U, Z, StrataEps, Eps)
        C = self.GenerateC(X, U, T)
        M = self.GenerateM(X, U, T, C)
        return X, Z, T, C, M, S
