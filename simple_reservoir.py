# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:12:44 2019

@author: JF
"""

import numpy as np
import networkx as nx
import json
import atexit
import os.path
from decimal import Decimal
from collections import OrderedDict
import datetime
# from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt

from scipy.integrate import odeint

def lorenz(X, t, a, b, c):
    '''
    X has the form: X =(x, y, z)
    So we can use (x, y, z) = X to get x, y and z respectively!
    dx/dt = a*(y - x)
    dy/dt = x*(b - z) - y
    dz/dt = x*y - c*z
    
    '''
    
    (x, y, z) = X
    dx = a*(y - x)
    dy = x*(b - z) - y
    dz = x*y - c*z
    
    return np.array([dx, dy, dz])
t = np.linspace(0,20, 10000)
lorenz_states = odeint(lorenz, (0, 1, 0), t, args = (10, 28, 3))


# if config file not exists, use this default config
default_config = """{
  "input": {
    "nodes": 2,
    "functions": 
      [
        "lambda x: np.sin(128 * np.pi * x)",
        "lambda x: x"
      ],
    "length": 5000
  },
  "reservoir": {
    "start_node": 95,
    "end_node": 100,
    "step": 2,
    "degree_function": "lambda x: np.sqrt(x)",
    "sigma": 0.5,
    "bias": 1,
    "leakage_rate": 0.3,
    "regression_parameter": 1e-8
  },
  "output": {
    "nodes": 2
  },
  "training": {
    "init": 1000,
    "train": 3000,
    "test": 2000,
    "error": 1000
  }
}"""



class MyReservoir:
    def __init__(self):
        # Input layer
        self.M = 1                      # 输入层节点
        self.input_len = 10000           # 输入长度
        # self.input_func = lambda x: np.sin(128 * np.pi * x)
        self.input_func = lambda x: lorenz_states[:,0]
        dataset = lorenz_states[:,0]
        # dataset.append(self.input_func(
        #         np.arange(self.input_len) / self.input_len))
        # self.dataset = lorenz_states[:,0]
        # print(dataset[1])
        self.dataset_in = np.array(lorenz_states[:,0])  # shape = (length , M)
        self.dataset_out = np.array(lorenz_states[:,1])
        # print(self.dataset.shape)

        # Reservoir layer
        self.N = 400   # 存储层节点个数
        self.degree_func = lambda x: np.sqrt(x)
        self.D = self.degree_func(self.N)
        self.sigma = 0.5
        self.bias = 1
        self.alpha = 0.3      # leakage rate
        self.beta = 1e-07     # 岭回归参数

        # Output layer
        self.P = 1

        # Training relevant
        self.init_len = 100
        self.train_len = 3000
        self.test_len = 1500
        # self.error_len = config["training"]["error"]


    def train(self):
        # 收集 reservoir state vectors  size = (N, train_len_len)
        self.r = np.zeros(
            (self.N, self.train_len - self.init_len))
        # 收集 input and output signals   size = (train_len - init_len, 1) 可改动
        self.u = np.vstack((self.dataset_in[i] for i in range(self.init_len, self.train_len))) 
        self.s = np.vstack((self.dataset_out[i] for i in range(self.init_len, self.train_len))) 
        
        # 设置随机数种子
        np.random.seed(42)

        # 初始化输入层与存储层权重矩阵
        self.Win = np.random.uniform(-self.sigma,
                                     self.sigma, (self.N, self.M))
        
        # 得到稀疏邻接矩阵 W  size = (N, N)
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, 42, True)
        # nx.draw(g, node_size=self.N)

        self.W = nx.adjacency_matrix(g).todense()
        # spectral radius: rho
        self.rho = max(abs(np.linalg.eig(self.W)[0]))
        self.W *= 1.25 / self.rho

        # run the reservoir with the data and collect r
        uu = np.vstack(self.dataset_in[t])
        rr = self.r[:,0]
        for t in range(self.train_len):    
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            rr = (1 - self.alpha) * self.rr + self.alpha * 
                np.tanh(np.dot(self.W, self.rr) + 
                np.dot(self.Win, uu) + self.bias*np.ones((self.N, 1)))
            if t >= self.init_len:
                self.r[:, [t - self.init_len]
                       ] = self.rr
        # train the output
        
        # Wout = (s * r^T) * ((r * r^T) + beta * I)
        self.Wout = np.dot(np.dot(self.S, R_T), np.linalg.inv(
            np.dot(self.R, R_T) + self.beta * np.eye(self.M + self.N + 1)))

    def _run(self):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        self.S = np.zeros((self.P, self.test_len))
        u = np.vstack((x[self.train_len] for x in self.dataset))
        for t in range(self.test_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,
                                                                             self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            s = np.dot(self.Wout, np.vstack((self.bias, u, self.r)))
            self.S[:, t] = np.squeeze(np.asarray(s))
            # use output as input
            u = s

    def draw(self):
      plt.subplots(1, self.M)
      plt.suptitle('N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
      for i in range(self.M):
        ax = plt.subplot(1, self.M, i + 1)
        plt.plot(self.S[i], label = 'prediction')
       # plt.plot(self.dataset[i][self.train_len + 1 : self.train_len + self.test_len + 1], label = 'input signal')
        plt.legend(loc = 'upper right')
        # plt.savefig('N = ' + str(self.N), dpi = 300)
      plt.show()
    
    def run(self):
        self.train()
        self._run()
        self.draw()


# Invoke automatically when exit, write the progress back to config file
def exit_handler():
    global config
    with open('reservoir.config', 'w') as config_file:
        config_file.write(json.dumps(config, indent = 4))
    print('Program finished! Current node = ' +
          str(config["reservoir"]["start_node"]))


if __name__ == '__main__':
   # atexit.register(exit_handler)
    r = MyReservoir()
    r.run()



    