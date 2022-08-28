#version 1

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.neighbors import KNeighborsClassifier as knn

# Set seed
np.random.seed(2)

p = 2
csize = 10
sigma = 1
m0_mean = [0,1]
m1_mean = [1,0]
cov = [[1,0],[0,1]]
n_sim = 50

m1 = np.random.normal(size = (csize, p)) * sigma + np.concatenate([np.array([[1, 0]] * csize)])
m0 = np.random.normal(size = (csize, p)) * sigma + np.concatenate([np.array([[0, 1]] * csize)]) # generate center m1

class sim_params :
    csize = 10           # number of centers
    p = 2                # dimension
    s = np.sqrt(1 / 5)   # standard deviation for generating data
    n = 100              # training size per class
    N = 5000             # test size per class
    m0 = m0              # 10 centers for class 0
    m1 = m1              # 10 centers for class 1

def generate_sim_data(sim_params):

    p = sim_params.p
    s = sim_params.s
    n = sim_params.n
    N = sim_params.N
    m1 = sim_params.m1
    m0 = sim_params.m0
    csize = sim_params.csize
    
    id1 = np.random.randint(csize, size = n)
    id0 = np.random.randint(csize, size = n)

    Xtrain = np.random.normal(size = (2 * n, p)) * s \
                + np.concatenate([m1[id1,:], m0[id0,:]])
    Ytrain = np.concatenate(([1]*n, [0]*n))

    id1 = np.random.randint(csize, size = N)
    id0 = np.random.randint(csize, size = N)
    Xtest = np.random.normal(size = (2 * N, p)) * s \
                + np.concatenate([m1[id1,:], m0[id0,:]])
    Ytest = np.concatenate(([1]*N, [0]*N))

    return Xtrain, Ytrain, Xtest, Ytest

Xtrain, Ytrain, Xtest, Ytest = generate_sim_data(sim_params)


XtrainSim = np.zeros((n_sim,2 * sim_params.n,sim_params.p))
YtrainSim = np.zeros((n_sim,2 * sim_params.n))
XtestSim = np.zeros((n_sim,2 * sim_params.N,sim_params.p))
YtestSim = np.zeros((n_sim,2 * sim_params.N))

for i in range(n_sim):
  XtrainSim[i], YtrainSim[i], XtestSim[i], YtestSim[i] = generate_sim_data(sim_params)


# new code
# x0 is a single 2D vector
# n0 is knn n searching parameter
def myknn(xtrain, ytrain, x0, n0):
  # matrix: first row = distance, second row = corresponding y training value
  matrix = np.vstack((LA.norm(xtrain-x0, axis=1).transpose(),
                      ytrain[np.newaxis])).transpose()
  # final: sorted matrix based on distance.
  final = matrix[matrix[:, 0].argsort()]
  # y_searched: y value list, from nearested to farest
  y_searched = final[:,1]
  #print(y_searched)
  # y_pred: average of first n nearest results
  y_pred = np.sum(y_searched[0:n0])/n0
  #print("y_prep is" ,n * y_pred)

  #selection rule
  if y_pred > 0.5:
    return 1
  else:
    return 0

myknn_predict = []

for i in range(2 * sim_params.N):
  myknn_predict.append(myknn(Xtrain,Ytrain,Xtest[i],3))


################################################################################

#define n-fold of cv-knn algorithm
nf = 10

#writing cv knn function
def cvknn(Xtrain, Ytrain, nf, k):
  cv_fold_number = int(2*sim_params.n /nf)

  cv_x = np.empty([2 * sim_params.n,2])
  cv_y = np.empty([2 * sim_params.n])

  cv_x[0::2] = Xtrain[0:sim_params.n]
  cv_x[1::2] = Xtrain[sim_params.n:2* sim_params.n]
  cv_y[0::2] = Ytrain[0:sim_params.n]
  cv_y[1::2] = Ytrain[sim_params.n:2* sim_params.n]

  cv_x_sets = np.zeros((nf,cv_fold_number * (nf-1),2))
  for i in range(nf):
      cv_x_sets[i] = np.delete(cv_x, i * cv_fold_number + np.array(range(cv_fold_number)),0)
      
  cv_y_sets = np.zeros((nf,cv_fold_number * (nf-1)))
  for i in range(nf):
      cv_y_sets[i] = np.delete(cv_y, i * cv_fold_number + np.array(range(cv_fold_number)))

  #myknn(cv_x[0:20],cv_y[0:20],Xtest[1],k)
  
  cv_error = 0
  result = np.array([myknn(cv_x_sets[i],cv_y_sets[i],element,k) for element in cv_x[i*cv_fold_number :(i+1)*cv_fold_number]]) 
  cv_error += np.sum(result != cv_y[0:20])/(2*nf)

  return cv_error


##################################################################################
#Bayes rule
def bayes(x):
  d1 = sum(np.exp(- ((m1[i, 0] - x[0]) ** 2 + (m1[i, 1] - x[1]) ** 2) / (2 * sim_params.s ** 2)) for i in range(len(m1)))
  d0 = sum(np.exp(- ((m0[i, 0] - x[0]) ** 2 + (m0[i, 1] - x[1]) ** 2) / (2 * sim_params.s ** 2)) for i in range(len(m0)))
  if d1/d0 > 1:
    return 1
  else:
    return 0


 
