import numpy as np
from mv_laplace import MvLaplaceSampler
from scipy.stats import logistic


class DataGenerator:
  def __init__(self,*, N = 1000, strata_size = 10, beta = 1, MaskRate = 0.3, model = 0, verbose = False, bias = False, Missing_lambda= None):
    self.N = N
    self.beta = beta
    self.strata_size = strata_size
    self.totalStrataNumber = int(N/strata_size)
    self.MaskRate = MaskRate
    self.verbose = verbose
    self.bias = bias
    self.model = model
    self.Missing_lambda = Missing_lambda
    
  
  def GenerateX(self):

      # generate Xn1 and Xn2
      mean = [1/2, -1/3]
      cov = [[1, 1/2], [1/2, 1]]
      X1_2 = np.random.multivariate_normal(mean, cov, self.N)

      # generate Xn3 and Xn4
      loc = [0, 1/np.sqrt(3)]
      cov = [[1,1/np.sqrt(2)], [1/np.sqrt(2),1]]

      sampler = MvLaplaceSampler(loc, cov)
      X3_4 = sampler.sample(self.N)

      # generate Xn5
      p = 1/3
      X5 = np.random.binomial(1, p, self.N)

      # combine all generated variables into a single matrix
      X = np.hstack((X1_2, X3_4, X5.reshape(-1,1)))
      
      return X

  def GenerateU(self):
      # generate U
      mean = 0
      std = np.sqrt(0.2)
      U = np.random.normal(mean, std, self.N)
      U = U.reshape(-1, 1)
      return U

  def GenerateS(self):
    # Add strata index
    S = np.zeros(self.N)
    for i in range(self.totalStrataNumber):
        S[self.strata_size*i:self.strata_size*(i+1)] = i + 1
    S = S.reshape(-1, 1)
    return S

  def GenerateZ(self):
    Z = []
    half_strata_size = self.strata_size // 2  # Ensure strata_size is even

    for i in range(self.totalStrataNumber):
        strata = np.array([0.0]*half_strata_size + [1.0]*half_strata_size)
        np.random.shuffle(strata)
        Z.append(strata)
    Z = np.concatenate(Z).reshape(-1, 1) 

    return Z


  def GenerateIndividualEps(self):
      mean = 0
      std = np.sqrt(0.2)
      eps = np.random.normal(mean, std, self.N)
      eps = eps.reshape(-1,)

      return eps
  
  def GenerateStrataEps(self):
      eps = []

      for i in range(self.totalStrataNumber):
          eps.append(np.full(self.strata_size, np.random.normal(0, np.sqrt(0.1))))

      eps = np.concatenate(eps).reshape(-1,)
      return eps
  
  def GenerateXInter(self, X):
      biases = []

      for i in range(self.totalStrataNumber):
          strata = X[i * self.strata_size : (i+1) * self.strata_size, 0]  # select the first column in the strata
          biases.append(np.full(self.strata_size, (10/3)*np.mean(strata)))

      biases = np.concatenate(biases).reshape(-1,)
      return biases

  def GenerateYInter(self, Y):
      biases = []
      for i in range(self.totalStrataNumber):
          strata = Y[i * self.strata_size : (i+1) * self.strata_size, 0]  # select the first column in the strata
          biases.append(np.full(self.strata_size, (10/3)*np.mean(strata)))

      biases = np.concatenate(biases).reshape(-1,)
      return biases

  def GenerateY(self, X, U, Z,  StrataEps, IndividualEps):
        
    sum1 = np.zeros(self.N)
    for p in range(1,6):
      sum1 += X[:,p-1]
    sum1 = (1.0 / np.sqrt(5)) * sum1

    sum2 = np.zeros(self.N)
    for p in range(1,6):
      for p_2 in range(1,6):
        sum2 += X[:,p-1] * logistic.cdf(1 - X[:,p_2-1])
    sum2 = (1.0 / np.sqrt(5 * 5)) * sum2

    sum3 = np.zeros(self.N)
    for p in range(1, 6):
        sum3 += X[:, p-1]**2
    sum3 = (1.0 / np.sqrt(5)) * sum3

    sum4 = np.zeros(self.N)
    for p in range(1,6):
      sum4 += np.absolute(X[:,p-1])
    sum4 = (1.0  / np.sqrt(5)) * sum4

    U = U.reshape(-1,)
    Z = Z.reshape(-1,)

    assert(self.model == 1 or self.model == 2 or self.model == 3 or self.model == 4 or self.model == 5 or self.model == 6)
    if self.model == 1:
      Y = self.beta * Z + sum1 + U +  StrataEps+ IndividualEps 
    if self.model == 2:
      Y = self.beta * Z  + sum1 + sum2 + U +  StrataEps+ IndividualEps 
    if self.model == 3:
      Y = self.beta * Z +  self.beta * Z * X[:,0 ] + self.beta * Z * sum4 + sum1 + sum2 + U +  StrataEps+ IndividualEps
    if self.model == 4:
      Y = self.beta * Z +  self.beta * Z * X[:,0 ] + self.beta * Z * sum4 + sum1 + sum2 + U +  StrataEps+ IndividualEps
    if self.model == 6:
      Y = self.beta * Z * X[:,0] ** 2 + sum3 + U +  StrataEps+ IndividualEps

    Y = Y.reshape(-1, 1)

    return Y


  def GenerateM(self, X, U, Y, XInter, YInter):
      
      U = U.reshape(-1,)
      n = X.shape[0]

      M = np.zeros((n, 1))
      M_lamda = np.zeros((n, 1))
      for i in range(n):
        sum1 = 0
        for p in range(1,6):
            sum1 += p * X[i,p-1] 
        sum1 = (1.0  / np.sqrt(5)) * sum1

        sum2 = 0
        for p in range(1,6):
          sum2 += p * np.cos(X[i,p-1])
        sum2 =  (1.0  / np.sqrt(5)) * sum2

        assert( self.model == 1 or self.model == 2 or self.model == 3 or self.model == 4 or self.model == 6) 
        if self.model == 1:
          M_lamda[i][0] = sum1 + Y[i, 0] + U[i] 
        if self.model == 2:
          M_lamda[i][0] = sum1 + sum2 + 10 * logistic.cdf(Y[i, 0]) + U[i] 
        if self.model == 3:
          M_lamda[i][0] = sum1 + sum2  + 10 * logistic.cdf(Y[i, 0]) + U[i]
        if self.model == 4:
          M_lamda[i][0] = sum1 + sum2  + 10 * logistic.cdf(Y[i, 0])+ XInter[i] + YInter[i] + U[i]
        if self.model == 6:
          M_lamda[i][0] = sum1 + sum2  + 10 * logistic.cdf(Y[i, 0])+ XInter[i] + YInter[i] + U[i]

      if self.Missing_lambda == None:
        lambda1 = np.percentile(M_lamda, 100 * (1-self.MaskRate))
      else:
        lambda1 = self.Missing_lambda

      for i in range(n):
        sum1 = 0
        for p in range(1,6):
            sum1 += p * X[i,p-1] 
        sum1 = (1.0  / np.sqrt(5)) * sum1

        sum2 = 0
        for p in range(1,6):
          sum2 += p * np.cos(X[i,p-1])
        sum2 =  (1.0  / np.sqrt(5)) * sum2

        assert(self.model == 1 or self.model == 2 or self.model == 3 or self.model == 4 or self.model == 6)
    
        if self.model == 1:
          if sum1 + Y[i, 0] + U[i]  > lambda1:
            M[i][0] = 1
        if self.model == 2:
          if sum1 + sum2   + 10 * logistic.cdf(Y[i, 0])+ U[i] > lambda1:
            M[i][0] = 1            
        if self.model == 3:
          if sum1 + sum2  + 10 * logistic.cdf(Y[i, 0]) + U[i] > lambda1:
            M[i][0] = 1 
        if self.model == 4:
          if sum1 + sum2  + 10 * logistic.cdf(Y[i, 0]) + XInter[i] + YInter[i] + U[i] > lambda1:
            M[i][0] = 1
        if self.model == 6:
          if sum1 + sum2  + 10 * logistic.cdf(Y[i, 0]) + XInter[i] + YInter[i] + U[i] > lambda1:
            M[i][0] = 1

      return M
  
  def GenerateData(self):
    X = self.GenerateX()
    Z = self.GenerateZ()
    U = self.GenerateU()
    S = self.GenerateS()
    IndividualEps = self.GenerateIndividualEps()
    StrataEps = self.GenerateStrataEps()
    XInter = self.GenerateXInter(X)
    YInter = self.GenerateYInter(X)
    Y = self.GenerateY(X, U, Z, StrataEps, IndividualEps)
    M = self.GenerateM(X, U, Y, XInter, YInter)
    return X, Z, U, Y, M, S

  