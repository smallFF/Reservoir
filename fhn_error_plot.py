# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:12:44 2019

@author: JF
"""

import numpy as np
import networkx as nx
import json
# import atexit
# import os.path
# from decimal import Decimal
# from collections import OrderedDict
# import datetime
# from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numba import jit
from matplotlib.ticker import FuncFormatter

# @jit(cache=True, nopython=True)
# def FHN(X, t, a, b, c, f):
#     (x, y) = X
#     dx = x * (x - 1)*(1 - b * x) - y + a / \
#               (2 * np.pi * f) * np.cos(2 * f*np.pi * t)
#     dy = c * x

#     return [dx, dy]

class Model:
    """
    :functions:
        initial model with some parameters

    :param:
        name: model name.
            Now only lorenz model and rossler model are supported!
        init_value, t, args:
            these three parameters are use to get states data
            states = odeint(func, init_value, t, args)
        N: the number of reservoir nodes
        rho: spectral radius
        sigma: scale of input weights
        bias: bias constant
        alpha: leakage rate
        beta: ridge regression parameter
        normalize:
            Default normalize = True
            if normalize param is True, we process the states into the form with zero mean and unit variance
            if normaliz param is False, we use the raw states data
        states:
            model states with the shpae (length, dimEquation)

    :return:
        no return
    """

    def __init__(self, name='lorenz', init_value=(0.1, 0.1, 0.1), t=(0, 50, 0.01), args=(10, 28, 8 / 3), N=400,
                 rho=0.9, sigma=1.0, bias=1.0, alpha=1.0, beta=1e-08, normalize=True):
        if name not in ('lorenz', 'rossler','fhn'):
            raise ValueError('Now noly \'lorenz\' model, \'rossler\' model and \'fhn\' model are supported!')
        else:
            self.name = name
            self.init_value = init_value
            self.t_config = t
            self.t = np.arange(self.t_config[0], self.t_config[1], self.t_config[2])
            self.args = args
            self.N = N
            self.rho = rho
            self.sigma = sigma
            self.bias = bias
            self.alpha = alpha
            self.beta = beta
            self.normalize = normalize

            fun_list = {'lorenz': self._lorenz, 'rossler': self._rossler, 'fhn': self._fhn}

            states = np.array(odeint(fun_list[name], self.init_value, self.t, args=self.args))
            if self.normalize:
                states = (states - np.mean(states, 0)) / np.mean((states - np.mean(states, 0)) ** 2, 0) ** (1 / 2)
            self.states = states

    # lorenz系统
    # @jit(cache=True, nopython=True)
    def _lorenz(self, X, t, a, b, c):
        """
        X has the form: X =(x, y, z)
        So we can use (x, y, z) = X to get x, y and z respectively!
        dx/dt = a*(y - x)
        dy/dt = x*(b - z) - y
        dz/dt = x*y - c*z
        """
        (x, y, z) = X
        dx = a * (y - x)
        dy = x * (b - z) - y
        dz = x * y - c * z
        return [dx, dy, dz]

    # rossler系统
    # @jit(cache=True, nopython=True)
    def _rossler(self, X, t, a, b, c):
        """
        X has the form: X =(x, y, z)
        So we can use (x, y, z) = X to get x, y and z respectively!
        dx/dt = - y - z 
        dy/dt = x + a*y
        dz/dt = b + z*(x-c)
        """
        (x, y, z) = X
        dx = - y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return [dx, dy, dz]
    
    # @jit(cache=True, nopython=True)
    def _fhn(self, X, t, a, b, c, f):
        (x, y) = X
        dx = x * (x - 1)*(1 - b * x) - y + a / \
                (2 * np.pi * f) * np.cos(2 * f*np.pi * t)
        dy = c * x

        # return FHN(X, t, a, b, c, f)
        return [dx, dy]

    def __str__(self):
        T0, T, delta_t = self.t_config
        model = {
            'model name': self.name,
            'init_value': str(self.init_value),
            't': 'np.arange(%s, %s ,%s)' % (T0, T, delta_t),
            'args': str(self.args),
            'N (number of reservoir nodes)': str(self.N),
            'rho (spectral radius)': str(self.rho),
            'sigma (scale of input weights)': str(self.sigma),
            'bias (bias constant)': str(self.bias),
            'alpha (leakage rate)': str(self.alpha),
            'beta (ridge regression parameter)': str(self.beta),
            'normalize (get zero mean and unit variance states)': str(self.normalize)
        }
        json_dumps_indent_str = json.dumps(model, indent=2)
        return json_dumps_indent_str

    __repr__ = __str__


class Reservoir:
    """
    :initial the reservoir
        input: the shape of input data should be (M, length)
        output: the shape of output data shoule be (P, lenght)
        model_obj: an instance of Model class
        rand_seed: random_seed, default value = 42
    :parameter
        model = model_obj (some parameters are included in the model_obj)
        dataset_in = input
        dataset_out = output
        intput_len: the number of all data
        train_len: We set the train_len approximately equals to 0.5 * input_len, which means training set has a half of all data
        u: training data
        r: reservoir states
        s: true output data
        S: predicted results
        Win: input weight
        W: reservoir layer adjacency matrix
        Wout: output weight, C: constans

        reservoir update equation:
            r(t+dt) = (1-alpha) * r(t) + alpha * tanh(W*r(t) + Win*u(t) + bias*I)

        predict equation:
            S = Wout * r + C

        RMS: rms error for all data
        RMS_partial: rms error for partial data
    """

    def __init__(self, input, output, model_obj, rand_seed=42):
        self.model = model_obj
        self.model_name = self.model.name
        self.random_seed = rand_seed

        self.dataset_in = input  # shape = (M,lenght)
        self.dataset_out = output  # sh]ape = (P,lenght)
        self.input_len = input.shape[1]  # 输入数据长度

        # print("dataset_in.shape = ", self.dataset_in.shape)
        # print("dataset_out.shape = ", self.dataset_out.shape)

        # Input layer
        self.M = self.dataset_in.shape[0]  # 输入层节点

        # Output layer
        self.P = self.dataset_out.shape[0]  # 输出层节点

        # Reservoir layer
        self.N = self.model.N  # 存储层节点个数
        self.degree_func = lambda x: np.sqrt(x)
        self.D = self.degree_func(self.N)
        self.sigma = self.model.sigma
        self.bias = self.model.bias
        self.alpha = self.model.alpha  # leakage rate
        self.beta = self.model.beta  # 岭回归参数

        self.RMS = []
        self.RMS_partial = []

        # TODO: Training relevant 可以删除self.init_len 和 self.error_len
        self.init_len = 0
        self.train_len = int(0.5 * self.input_len)
        self.test_len = 0
        # self.error_len = config["training"]["error"]

        # 初始化各种状态
        # 收集 reservoir state vectors  size = (N, train_len - init_len)
        self.r = np.zeros(
            (self.N, self.train_len - self.init_len))
        # print("self.r.shape = ", self.r.shape)

        # 收集 input signals   size = (M,train_len - init_len) 可改动
        self.u = self.dataset_in[:, self.init_len: self.train_len]
        # print("self.u.shape = ", self.u.shape)

        # 收集 output signals   size = (P,train_len - init_len) 可改动
        self.s = self.dataset_out[:, self.init_len: self.train_len]
        # print("self.s.shape = ", self.s.shape)

        # 记录整个数据集的输出
        self.S = np.zeros((self.P, self.input_len))
        # 设置随机数种子
        np.random.seed(self.random_seed)

        # 初始化输入层与存储层权重矩阵
        self.Win = np.random.uniform(-self.sigma, self.sigma, (self.N, self.M))
        # print("self.Win.shape = ", self.Win.shape)

        # 得到稀疏邻接矩阵 W  size = (N, N)
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        # g = nx.erdos_renyi_graph(self.N, self.D / self.N, self.random_seed, True)
        # # nx.draw(g, node_size=self.N)
        # self.W = nx.adjacency_matrix(g).todense()
        # print("self.W.shape = ",self.W.shape)

        # self.W = np.random()
        self.W = np.random.uniform(-1, 1, (self.N, self.N))

        # spectral radius: rho  谱半径
        self.rho = max(np.abs(np.linalg.eig(self.W)[0]))
        # print("self.rho = ", self.rho)
        self.W *= self.model.rho / self.rho

        self.Wout = []
        self.C = []

    def train(self):
        """
        :function:
            run the reservoir with the data and collect r
            get Wout and C

        :return:
            No return
        """
        rr = np.zeros((self.N, 1))
        # print("rr.shape = ", rr.shape)
        for t in range(self.train_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            uu = self.u[:, [t]]
            rr = (1 - self.alpha) * rr + self.alpha * \
                 np.tanh(np.dot(self.W, rr) + \
                         np.dot(self.Win, uu) + self.bias * np.ones((self.N, 1)))
            #            print("uu.shape = ", uu.shape)
            #            print("rr.shape = ", rr.shape)

            if t >= self.init_len:
                self.r[:, [t - self.init_len]] = rr

        # train the output 
        # Wout = (s * r^T) * ((r * r^T) + beta * I)  这个公式好像不太对

        # 得到s_mean 和 r_mean
        s_mean = 0
        r_mean = 0

        for i in range(self.init_len, self.train_len):
            s_mean += self.s[:, [i - self.init_len]]
            r_mean += self.r[:, [i - self.init_len]]
        s_mean /= self.train_len
        r_mean /= self.train_len
        # print("s_mean = ", s_mean)
        # print("r_mean = ", r_mean)

        delta_s = np.zeros(self.s.shape)
        delta_r = np.zeros(self.r.shape)
        # print("delta_s.shape = ", delta_s.shape)
        # print("delta_r.shape = ", delta_r.shape)
        for i in range(self.train_len - self.init_len):
            delta_s[:, [i]] = self.s[:, [i]] - s_mean
            delta_r[:, [i]] = self.r[:, [i]] - r_mean

        self.Wout = np.dot(np.dot(delta_s, delta_r.T), np.linalg.inv(
            np.dot(delta_r, delta_r.T) + self.beta * np.eye(self.N)))
        self.C = -(np.dot(self.Wout, r_mean) - s_mean)

    def predict(self, partial=False):
        """
        :function:
            get S and RMS error or RMS_partial error

        :param partial:
            Default partial is False
            If partial is False, we just predict the data after train_len, which means dataset_out[train_len, input_len]
            If partial is True, we predict the data of all dataset_out, which means dataset_out[0, input_len]

        :return:
            No return
        """
        if not partial:
            # 我选择整个数据集做训练
            rr = np.zeros((self.N, 1))

            # print("rr.shape = ", rr.shape)
            for t in range(self.input_len):
                # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
                uu = self.dataset_in[:, [t]]
                rr = (1 - self.alpha) * rr + self.alpha * \
                     np.tanh(np.dot(self.W, rr) + \
                             np.dot(self.Win, uu) + self.bias * np.ones((self.N, 1)))

                s = np.dot(self.Wout, rr) + self.C
                self.S[:, [t]] = s

            # print(self.S)
            # print("self.S.shape = ", self.S.shape)

            # 计算RMS error
            for i in range(0, self.P):
                self.RMS.append(np.sqrt(np.sum((self.dataset_out[i] - self.S[i]) ** 2 / self.input_len)))
            # print('RMS error = %s' % self.RMS)
        else:
            rr = self.r[:, [self.train_len - 1]]
            for t in range(self.train_len, self.input_len):
                uu = self.dataset_in[:, [t]]
                rr = (1 - self.alpha) * rr + self.alpha * \
                     np.tanh(np.dot(self.W, rr) + \
                             np.dot(self.Win, uu) + self.bias * np.ones((self.N, 1)))

                s = np.dot(self.Wout, rr) + self.C
                self.S[:, [t]] = s

            # 计算RMS error partial
            for i in range(0, self.P):
                self.RMS_partial.append(np.sqrt(np.sum((self.dataset_out[[i], self.train_len:self.input_len]
                                                        - self.S[[i], self.train_len:self.input_len]) ** 2 / (
                                                               self.input_len - self.train_len))))

    # 为了展示两个模型的结果，注释掉Reservoir类中draw函数的plt.show()功能，结果见生成的图片
    def draw(self, partial=False):
        """
        :function:
            show the results and save the results in the form of pictures
        :param partial:
            Default partial is False
            If partial is False, we just draw the data after train_len, which means dataset_out[train_len, input_len]
        and S[train_len, input_len]
            If partial is True, we draw the data of all dataset_out, which means dataset_out[0, input_len] and
        S[0, input_len]

        :return:
            no return
        """
        # 根据模型获取时间数据
        t = self.model.t
        if not partial:
            plt.subplots(self.P, 1)
            # plt.suptitle(self.model_name +', N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
            plt.suptitle('%s N = %s dt = %s' % (self.model_name, str(self.N), self.model.t_config[2]))
            plt.subplots_adjust(hspace=1)

            for i in range(self.P):
                ax = plt.subplot(self.P, 1, i + 1)
                plt.text(0.5, -0.3, 'RMS error = %s' % self.RMS[i], size=10, ha='center', transform=ax.transAxes)
                plt.plot(t, self.dataset_out[i], ls='-', label='true output signal')
                plt.plot(t, self.S[i], ls='--', label='prediction')
                plt.legend(loc='upper right')
            plt.savefig('Python--%s N = %s dt = %s.png' % (self.model_name, self.N, self.model.t_config[2]), dpi=600)
            plt.show()
        else:
            plt.subplots(self.P, 1)
            # plt.suptitle(self.model_name +', N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
            plt.suptitle('%s N = %s dt = %s partial' % (self.model_name, str(self.N), self.model.t_config[2]))
            plt.subplots_adjust(hspace=1)

            for i in range(self.P):
                ax = plt.subplot(self.P, 1, i + 1)
                plt.text(0.5, -0.3, 'RMS error partial = %s' % self.RMS_partial[i], size=10, ha='center',
                         transform=ax.transAxes)
                partial_t = t[self.train_len: self.input_len]
                partial_dataset_out = self.dataset_out[:, self.train_len: self.input_len]
                partial_S = self.S[:, self.train_len: self.input_len]
                plt.plot(partial_t, partial_dataset_out[i], ls='-', label='true output signal')
                plt.plot(partial_t, partial_S[i], ls='--', label='prediction partial')
                plt.legend(loc='upper right')
            plt.savefig('Python--%s N = %s dt = %s partial.png' % (self.model_name, self.N, self.model.t_config[2]),
                        dpi=600)
            plt.show()

    # 集训练、预测和绘图功能于一体
    def run(self, partial=False):
        self.train()
        self.predict(partial)
        self.draw(partial)

# def draw_error_plot(model_name='fhn',ind_x=[0], ind_y=[1],
#                     init_value=(0, 1.0),
#                     t=(0, 400, 0.01),
#                     args=(0.1, 10, 1.0, 0.13),
#                     N=400,
#                     rho=0.94,
#                     sigma=1.0,
#                     bias=1.0,
#                     alpha=1.0,
#                     beta=1e-08,
#                     normalize=False,partial=True):
#     model = Model(model_name, init_value=init_value, t=t, args=args,
#                 sigma=sigma, bias=bias, alpha=alpha, beta=beta, normalize=normalize)
#     input = model.states[:,ind_x].T
#     output = model.states[:,ind_y].T
#     r = Reservoir(input, output, model)
#     r.train()
#     r.predict(partial)
#     return r.RMS_partial

# def draw_specific(par, par_set, xlabel, fontdict):
    
#     rms_error = []
#     for _ in par_set:


if __name__ == '__main__':
    # 目前模型的选择只有两个，分别是 lorenz 和 rossler.
    fontdict = {'family': 'DejaVu Sans' ,'size': 10}
    f = 0.13

    # rho optimal=1.79
    rho_set = np.linspace(0.8, 1.9, 20)
    rms_error = []
    for rho in rho_set:
        model_1 = Model('fhn', init_value=(0, 1.0), 
                        t=(0, 400, 0.01), args=(0.1, 10, 1.0, f),
                        N=400, rho=rho,
                        sigma=1.0, bias=1.0, alpha=0.66, 
                        beta=1e-08, normalize=False)
        # print(model_1)
        input1 = model_1.states[:, [0]].T
        output1 = model_1.states[:, [1]].T
        r1 = Reservoir(input1, output1, model_1)
        r1.train()
        r1.predict(partial=True)
        rms_error.append(r1.RMS_partial)
        # plt.figure()
        # r1.draw(partial=True)
    rms_error = np.array(rms_error)
    plt.figure()
    plt.plot(rho_set, rms_error, 'bo', markersize=3)
    plt.errorbar(rho_set, rms_error, yerr=rms_error*0.035, elinewidth=1, capsize=3)
    plt.xlabel(r'$\rho$', fontdict=fontdict)
    plt.ylabel('RMS error', fontdict=fontdict)
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.show()
    

    # alpha optimal=0.66
    alpha_set = np.linspace(0.01, 1, 20)
    par_set = alpha_set
    xlabel = r'$\alpha$'
    rms_error = []
    for par in par_set:
        model_1 = Model('fhn', init_value=(0, 1.0),
                        t=(0, 400, 0.01),
                        args=(0.1, 10, 1.0, f), N=400, rho=1.75,
                        sigma=1.0, bias=1.0, alpha=par,
                        beta=1e-08, normalize=False)
        # print(model_1)
        input1 = model_1.states[:, [0]].T
        output1 = model_1.states[:, [1]].T
        r1 = Reservoir(input1, output1, model_1)
        r1.train()
        r1.predict(partial=True)
        rms_error.append(r1.RMS_partial)
        # plt.figure()
        # r1.draw(partial=True)
    rms_error = np.array(rms_error)
    plt.figure()
    plt.plot(par_set, rms_error, 'bo', markersize=3)
    plt.errorbar(par_set, rms_error, yerr=rms_error*0.035, elinewidth=1, capsize=3)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel('RMS error', fontdict=fontdict)
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.show()

    # sigma optimal=1
    sigma_set = np.linspace(0.01, 1.5, 20)
    par_set = sigma_set
    xlabel = r'$\sigma$'
    rms_error = []
    for par in par_set:
        model_1 = Model('fhn', init_value=(0, 1.0),
                        t=(0, 400, 0.01),
                        args=(0.1, 10, 1.0, f), N=400, rho=1.75,
                        sigma=par, bias=1.0, alpha=0.66,
                        beta=1e-08, normalize=False)
        # print(model_1)
        input1 = model_1.states[:, [0]].T
        output1 = model_1.states[:, [1]].T
        r1 = Reservoir(input1, output1, model_1)
        r1.train()
        r1.predict(partial=True)
        rms_error.append(r1.RMS_partial)
        # plt.figure()
        # r1.draw(partial=True)
    rms_error = np.array(rms_error)
    plt.figure()
    plt.plot(par_set, rms_error, 'bo', markersize=4)
    plt.errorbar(par_set, rms_error, yerr=rms_error*0.035, elinewidth=1, capsize=3)
    plt.plot(par_set, rms_error)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel('RMS error', fontdict=fontdict)
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.show()

    # N optimal=400
    N_set = [5*2**i for i in range(1,10)]
    par_set = N_set
    xlabel = r'$N$'
    rms_error = []
    for par in par_set:
        model_1 = Model('fhn', init_value=(0, 1.0),
                        t=(0, 400, 0.01),
                        args=(0.1, 10, 1.0, f), N=par, rho=1.75,
                        sigma=1, bias=1.0, alpha=0.66,
                        beta=1e-08, normalize=False)
        # print(model_1)
        input1 = model_1.states[:, [0]].T
        output1 = model_1.states[:, [1]].T
        r1 = Reservoir(input1, output1, model_1)
        r1.train()
        r1.predict(partial=True)
        rms_error.append(r1.RMS_partial)
        # plt.figure()
        # r1.draw(partial=True)
    rms_error = np.array(rms_error)
    plt.figure()
    plt.plot(par_set, rms_error, 'bo', markersize=3)
    plt.errorbar(par_set, rms_error, yerr=rms_error*0.035, elinewidth=1, capsize=3)
    plt.xlabel(xlabel, fontdict=fontdict)
    plt.ylabel('RMS error', fontdict=fontdict)
    ax = plt.gca()
    ax.yaxis.get_major_formatter().set_powerlimits((0,1))
    plt.show()