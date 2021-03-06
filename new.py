import numpy as np
import math
import networkx as nx
import json
import matplotlib.pyplot as plt

from scipy.integrate import odeint


class Model:
    '''
    生成实例有三个属性：name, model, states
    name:   字符串格式
    model:  字典格式，里面存储模型所需要各种参数
    states: np.array格式， shape = (length, n)
    '''

    def __init__(self, name):
        if name not in ('lorenz', 'rossler', 'new'):
            raise ValueError('Now only \'lorenz\' model and \'rossler\' model and \'new\' are supported!')
        else:
            # TODO: 现在只能分别设置时间和其他参数，下面时间格式为 t = np.arange(T0, T, delta_t)，后续待改进
            self.__time = {'lorenz': {'T0': 0, 'T': 100, 'delta_t': 0.01},
                           'rossler': {'T0': 0, 'T': 360, 'delta_t': 0.01},
                           'new': {'T0': 100, 'T': 400, 'delta_t': 0.05}}
            model_list = {
                'lorenz': {
                    'init_value': (0.1, 0.1, 0.1),
                    't': np.arange(self.__time[name]['T0'], self.__time[name]['T'], self.__time[name]['delta_t']),
                    'args': (10, 28, 8 / 3),
                    'func': self.__lorenz,
                    'N': 400,
                    'rho': 1.25,
                    'sigma': 1.0,
                    'bias': 1.0,
                    'alpha': 0.94,
                    'beta': 1e-7,
                    'T0': self.__time[name]['T0'],
                    'T': self.__time[name]['T'],
                    'delta_t': self.__time[name]['delta_t']
                },
                'rossler': {
                    'init_value': (0.2, 1.0, -0.8),
                    't': np.arange(self.__time[name]['T0'], self.__time[name]['T'], self.__time[name]['delta_t']),
                    'args': (0.5, 2.0, 4.0),
                    'func': self.__rossler,
                    'N': 400,
                    'rho': 1.25,
                    'sigma': 1.0,
                    'bias': 1.0,
                    'alpha': 0.94,
                    'beta': 1e-8,
                    'T0': self.__time[name]['T0'],
                    'T': self.__time[name]['T'],
                    'delta_t': self.__time[name]['delta_t']
                },
                'new': {
                    'init_value': (0, 0),
                    't': np.arange(self.__time[name]['T0'], self.__time[name]['T'], self.__time[name]['delta_t']),
                    'args': (0.1, 10, 1.0),
                    'func': self.__new,
                    'N': 400,
                    'rho': 1.25,
                    'sigma': 1.0,
                    'bias': 1.0,
                    'alpha': 0.94,
                    'beta': 1e-8,
                    'T0': self.__time[name]['T0'],
                    'T': self.__time[name]['T'],
                    'delta_t': self.__time[name]['delta_t']
                }

            }
            self.name = name
            self.model = model_list[name]
            states = np.array(
                odeint(self.model['func'], self.model['init_value'], self.model['t'], args=self.model['args']))
            # states = (states - np.mean(states, 0)) / np.mean((states - np.mean(states, 0)) ** 2, 0) ** (1 / 2)
            self.states = states

    # lorenz系统
    def __lorenz(self, X, t, a, b, c):
        '''
        X has the form: X =(x, y, z)
        So we can use (x, y, z) = X to get x, y and z respectively!
        dx/dt = a*(y - x)
        dy/dt = x*(b - z) - y
        dz/dt = x*y - c*z
        '''
        (x, y, z) = X
        dx = a * (y - x)
        dy = x * (b - z) - y
        dz = x * y - c * z
        return np.array([dx, dy, dz])

    # rossler系统
    def __rossler(self, X, t, a, b, c):
        '''
        X has the form: X =(x, y, z)
        So we can use (x, y, z) = X to get x, y and z respectively!
        dx/dt = - y - z
        dy/dt = x + a*y
        dz/dt = b + z*(x-c)
        '''
        (x, y, z) = X
        dx = - y - z
        dy = x + a * y
        dz = b + z * (x - c)
        return np.array([dx, dy, dz])

    def __new(self, X, t, a, b, c):
        (x, y) = X
        dx = x * (x - 1)*(1 - b * x) - y + a / (2 * np.pi * 0.125)* np.cos(2 * 0.125*np.pi* t)
        dy = c * x
        return np.array([dx, dy])

    def __str__(self):
        name = self.name
        t = self.__time
        model_list = {
            name: {
                'init_value': str(self.model['init_value']),
                't': 'np.arange(%s, %s ,%s)' % (t[name]['T0'], t[name]['T'], t[name]['delta_t']),
                'args': str(self.model['args']),
                'func': '__%s in class Model' % name,
                'N': str(self.model['N']),
                'rho': str(self.model['rho']),
                'sigma': str(self.model['sigma']),
                'bias': str(self.model['bias']),
                'alpha': str(self.model['alpha']),
                'beta': str(self.model['beta']),
                'T0': str(self.model['T0']),
                'T': str(self.model['T']),
                'delta_t': str(self.model['delta_t'])
            }
        }
        jsonDumpsIndentStr = json.dumps(model_list, indent=2)
        return jsonDumpsIndentStr

    __repr__ = __str__


class Reservoir:
    def __init__(self, Model):
        self.model_name = Model.name
        self.model = Model.model
        self.random_seed = 42

        # 获取状态数据
        states = Model.states

        self.input_len = len(states)  # 输入长度

        self.dataset_in = states[:, [0]].T  # shape = (M,lenght)
        self.dataset_out = states[:, [1]].T  # sh]ape = (P,lenght)
        print(self.dataset_in)
        print(self.dataset_out)
        print("dataset_in.shape = ", self.dataset_in.shape)
        print("dataset_out.shape = ", self.dataset_out.shape)

        # Input layer
        self.M = self.dataset_in.shape[0]  # 输入层节点

        # Output layer
        self.P = self.dataset_out.shape[0]  # 输出层节点

        # Reservoir layer
        self.N = self.model['N']  # 存储层节点个数
        self.degree_func = lambda x: np.sqrt(x)
        self.D = self.degree_func(self.N)
        self.sigma = self.model['sigma']
        self.bias = self.model['bias']
        self.alpha = self.model['alpha']  # leakage rate
        self.beta = self.model['beta']  # 岭回归参数

        # TODO: Training relevant 可以删除self.init_len 和 self.error_len
        self.init_len = 0
        self.train_len = int(0.5 * self.input_len)
        self.test_len = 3000
        # self.error_len = config["training"]["error"]

        # 初始化各种状态
        # 收集 reservoir state vectors  size = (N, train_len - init_len)
        self.r = np.zeros(
            (self.N, self.train_len - self.init_len))
        print("self.r.shape = ", self.r.shape)

        # 收集 input signals   size = (M,train_len - init_len) 可改动
        self.u = self.dataset_in[:, self.init_len: self.train_len]
        print("self.u.shape = ", self.u.shape)

        # 收集 output signals   size = (P,train_len - init_len) 可改动
        self.s = self.dataset_out[:, self.init_len: self.train_len]
        print("self.s.shape = ", self.s.shape)

        # 设置随机数种子
        np.random.seed(self.random_seed)

        # 初始化输入层与存储层权重矩阵
        self.Win = np.random.uniform(-self.sigma, self.sigma, (self.N, self.M))
        print("self.Win.shape = ", self.Win.shape)

        # 得到稀疏邻接矩阵 W  size = (N, N)
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        # g = nx.erdos_renyi_graph(self.N, self.D / self.N, self.random_seed, True)
        # # nx.draw(g, node_size=self.N)
        # self.W = nx.adjacency_matrix(g).todense()
        # print("self.W.shape = ",self.W.shape)

        self.W = np.random.uniform(-1, 1, (self.N, self.N))

        # spectral radius: rho  谱半径
        self.rho = max(np.abs(np.linalg.eig(self.W)[0]))
        print("self.rho = ", self.rho)
        self.W *= self.model['rho'] / self.rho

    def train(self):
        # run the reservoir with the data and collect r
        uu = self.dataset_in[:, [0]]
        print(uu)
        print("uu.shape = ", uu.shape)
        rr = self.r[:, [0]]
        print("rr.shape = ", rr.shape)
        for t in range(self.train_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            uu = self.dataset_in[:, [t]]
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
        print("delta_s.shape = ", delta_s.shape)
        print("delta_r.shape = ", delta_r.shape)
        for i in range(self.train_len - self.init_len):
            delta_s[:, [i]] = self.s[:, [i]] - s_mean
            delta_r[:, [i]] = self.r[:, [i]] - r_mean

        self.Wout = np.dot(np.dot(delta_s, delta_r.T), np.linalg.inv(
            np.dot(delta_r, delta_r.T) + self.beta * np.eye(self.N)))
        self.C = -(np.dot(self.Wout, r_mean) - s_mean)

    def __run(self):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        # 我选择整个数据集
        self.S = np.zeros((self.P, self.input_len))
        uu = self.dataset_in[:, [0]]

        # 能直接使用前面的 r 吗？
        # TODO: 从这里开始错了啊 QAQ
        # rr = self.r[:,[self.train_len-self.init_len -1]]
        rr = np.zeros((self.N, 1))
        print("uu.shape = ", uu.shape)
        print("rr.shape = ", rr.shape)
        for t in range(self.input_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            # rr = (1 - self.alpha) * rr + self.alpha * np.tanh(np.dot(self.A,
            #       self.r) + np.dot(self.Win, np.vstack((self.bias, u))))

            #    print("uu.shape = ", uu.shape)
            #    print("rr.shape = ", rr.shape)

            uu = self.dataset_in[:, [t]]
            rr = (1 - self.alpha) * rr + self.alpha * \
                 np.tanh(np.dot(self.W, rr) + \
                         np.dot(self.Win, uu) + self.bias * np.ones((self.N, 1)))

            #    print("uu.shape = ", uu.shape)
            #    print("rr.shape = ", rr.shape)

            s = np.dot(self.Wout, rr) + self.C
            # print("s.shape = ", s.shape)
            self.S[:, [t]] = s
            # use output as input
            # 不能这么做
            # uu = s
        print(self.S)
        print("self.S.shape = ", self.S.shape)

        # 计算RMS error
        self.RMS = []
        for i in range(0, self.P):
            self.RMS.append(np.sqrt(np.sum((self.dataset_out[i] - self.S[i]) ** 2 / self.input_len)))
        # print('RMS error = %s' % self.RMS)

    def draw(self):
        # 根据模型获取时间数据
        t = self.model['t']
        """
        plt.subplots(2, 1)

        # plt.suptitle(self.model_name + ', N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
        plt.subplots_adjust(hspace=1)
        ax = plt.subplot(2, 1, 1)
        plt.plot(t-50, self.dataset_in[0])
        plt.xlabel('t')
        plt.ylabel('x')
        ax = plt.subplot(2, 1, 2)
        plt.plot(self.dataset_in[0][1000:4000],self.dataset_out[0][1000:4000])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        """

        plt.subplots(3, 1)
        plt.suptitle(self.model_name + ', N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
        plt.subplots_adjust(hspace=1)
        ax = plt.subplot(3, 1, 1)
        plt.text(0.5, -0.66, 'u(t) = x(t)', size=10, ha='center', transform=ax.transAxes)
        plt.plot(t, self.dataset_in[0])
        plt.xlabel('时间t')
        plt.ylabel('x(t)')

        # plt.show()

        # plt.show()

        for i in range(self.P):
            plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
            plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
            ax = plt.subplot(3, 1, 3)
            plt.text(0.5, -0.66, 'RMS error = %s' % self.RMS[i], size=10, ha='center', transform=ax.transAxes)
    
            plt.plot(t, self.dataset_out[i], ls='-', label='真实值')
            plt.plot(t, self.S[i], ls='--', label='预测值')
            plt.xlabel('时间t')
            plt.ylabel('y(t)')
            plt.legend(bbox_to_anchor=(0.81, 0.95), loc=3)


            # plt.legend(loc='upper right')


            ax = plt.subplot(3, 1, 2)
            plt.plot(self.dataset_in[0][1000:5000], self.dataset_out[0][1000:5000])
            plt.xlabel('x(t)')
            plt.ylabel('y(t)')
        # plt.savefig('%s N = %s.png' % (self.model_name, self.N), dpi=600)

        plt.show()

    # 集训练、预测和绘图于一体
    def run(self):
        self.train()
        self.__run()
        self.draw()


# 为了展示两个模型的结果，注释掉Reservoir类中draw函数的plt.show()功能，结果见生成的图片
# TODO: 后续修改成可以一次生成4-5个系统的结果
if __name__ == '__main__':
    # 目前模型的选择只有两个，分别是 lorenz 和 rossler.
    model_3 = Model('new')
    print(model_3)
    r3 = Reservoir(model_3)
    r3.run()
"""
 model_1 = Model('lorenz')
 print(model_1)
 r1 = Reservoir(model_1)
 r1.run()

 model_2 = Model('rossler')
 print(model_2)
 r2 = Reservoir(model_2)
 r2.run()
"""
