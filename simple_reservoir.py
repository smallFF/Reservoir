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

lorenz_states = (lorenz_states - np.mean(lorenz_states,0)) / np.mean((lorenz_states - np.mean(lorenz_states,0))**2,0)**(1/2)    


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
        print(dataset[1])
        self.dataset_in = np.vstack(lorenz_states[:,0]) # shape = (length , M)
        self.dataset_out = np.vstack(lorenz_states[:,1])
        print("dataset_in.shape = ", self.dataset_in.shape)
        print("dataset_out.shape = ", self.dataset_out.shape)

        # Reservoir layer
        self.N = 400   # 存储层节点个数
        self.degree_func = lambda x: np.sqrt(x)
        self.D = self.degree_func(self.N)
        self.sigma = 1
        self.bias = 1
        self.alpha = 1      # leakage rate
        self.beta = 1e-07     # 岭回归参数

        # Output layer
        self.P = 1

        # Training relevant
        self.init_len = 0
        self.train_len = 7000
        self.test_len = 3000
        # self.error_len = config["training"]["error"]


    def train(self):
        # 收集 reservoir state vectors  size = (N, train_len - init_len)
        self.r = np.zeros(
            (self.N, self.train_len - self.init_len))
        print("self.r.shape = ", self.r.shape)
        # 收集 input signals   size = (M,train_len - init_len) 可改动
        self.u = np.array([self.dataset_in[i] for i in range(self.init_len, self.train_len)]).T
        print("self.u.shape = ", self.u.shape)
        # 收集 output signals   size = (P,train_len - init_len) 可改动
        self.s = np.array([self.dataset_out[i] for i in range(self.init_len, self.train_len)]).T
        print("self.s.shape = ", self.s.shape)
        
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
        # spectral radius: rho  谱半径
        self.rho = max(abs(np.linalg.eig(self.W)[0]))
        print("self.rho = ", self.rho)
        self.W *= 1.0 / self.rho

        # run the reservoir with the data and collect r
        uu = np.vstack(self.dataset_in[0])
        print(uu)
        print("uu.shape = ", uu.shape)
        rr = np.vstack(self.r[:,0])
        print("rr.shape = ", rr.shape)
        for t in range(self.train_len):    
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            
            # term1 = (1 - self.alpha) * rr
            # print("term1.shape = ", term1.shape) 
            # term2 = self.alpha * \
            #     np.tanh(np.dot(self.W, rr) + \
            #     np.dot(self.Win, uu) + self.bias*np.ones((self.N, 1)))
            # print("term2.shape = ",term2.shape)
            uu = np.vstack(self.dataset_in[t])
            rr = (1 - self.alpha) * rr + self.alpha * \
                np.tanh(np.dot(self.W, rr) + \
                np.dot(self.Win, uu) + self.bias*np.ones((self.N, 1)))
            # print("rr_new.shape = ", rr.shape)
            if t >= self.init_len:
                self.r[:, [t - self.init_len]] = rr

        # train the output
        # Wout = (s * r^T) * ((r * r^T) + beta * I)
        # 得到s_mean 和 r_mean
        s_mean = 0
        r_mean = 0
        
        for i in range(self.init_len, self.train_len):
          s_mean += self.s[:,[i-self.init_len]]
          r_mean += self.r[:,[i-self.init_len]]
        s_mean /= self.train_len
        r_mean /= self.train_len
        # print("s_mean = ", s_mean)
        # print("r_mean = ", r_mean)

        delta_s = np.zeros(self.s.shape)
        delta_r = np.zeros(self.r.shape)
        print("delta_s.shape = ", delta_s.shape)
        print("delta_r.shape = ", delta_r.shape)
        for i in range(self.train_len - self.init_len):
          delta_s[:, [i]]= self.s[:, [i]] - s_mean
          delta_r[:, [i]] = self.r[:, [i]] - r_mean
          # print(delta_r[:,[i]])

        # delta_s = np.array(delta_s)
        # delta_r = np.array(delta_r)

        self.Wout = np.dot(np.dot(delta_s, delta_r.T), np.linalg.inv(
            np.dot(delta_r, delta_r.T) + self.beta * np.eye(self.N )))
        self.C = -(np.dot(self.Wout, r_mean) - s_mean)

    def _run(self):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        self.S = np.zeros((self.P, self.test_len))
        uu = np.vstack(self.dataset_in[self.train_len])

        # 能直接使用前面的 r 吗？
        # rr = self.r[:,[self.train_len-self.init_len -1]]
        rr = np.zeros((self.N,1))
        print("rr.shape = ", rr.shape)
        for t in range(self.test_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            # rr = (1 - self.alpha) * rr + self.alpha * np.tanh(np.dot(self.A,
            #                                                                  self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            uu = np.vstack(self.dataset_in[self.train_len + t])
            rr = (1 - self.alpha) * rr + self.alpha * \
                np.tanh(np.dot(self.W, rr) + \
                np.dot(self.Win, uu) + self.bias*np.ones((self.N, 1)))


            s = np.dot(self.Wout, rr) + self.C
            self.S[:, [t]] = np.squeeze(np.asarray(s))
            # use output as input
            # 不能这么做
            # uu = s
        print(self.S)
    def draw(self):
      t = np.linspace(0, 20, self.input_len)
      plt.subplots(1, self.M)
      plt.suptitle('N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
      plt.plot(t[self.train_len: self.train_len + self.test_len], self.S[0], label = 'prediction')
      plt.plot(t[self.train_len: self.train_len + self.test_len],
       self.dataset_out[self.train_len: self.train_len + self.test_len], label = 'true output signal')
      plt.legend(loc = 'upper right')
      # plt.savefig('N = ' + str(self.N), dpi = 300)
      plt.show()
    
    def run(self):
        self.train()
        self._run()
        self.draw()


# # Invoke automatically when exit, write the progress back to config file
# def exit_handler():
#     global config
#     with open('reservoir.config', 'w') as config_file:
#         config_file.write(json.dumps(config, indent = 4))
#     print('Program finished! Current node = ' +
#           str(config["reservoir"]["start_node"]))


if __name__ == '__main__':
   # atexit.register(exit_handler)
    r = MyReservoir()
    r.run()



    