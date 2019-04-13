
# def lorenz(X, t, a, b, c):
#     '''
#     X has the form: X =(x, y, z)
#     So we can use (x, y, z) = X to get x, y and z respectively!
#     dx/dt = a*(y - x)
#     dy/dt = x*(b - z) - y
#     dz/dt = x*y - c*z
    
#     '''
    
#     (x, y, z) = X
#     dx = a*(y - x)
#     dy = x*(b - z) - y
#     dz = x*y - c*z
#     return np.array([dx, dy, dz])

# def rossler(X, t, a, b, c):
#     (x, y, z) = X
#     dx = - y - z 
#     dy = x + a*y
#     dz = b + z*(x-c)
#     return np.array([dx, dy, dz])

# time = {'lorenz': np.arange(0, 50, 0.05), 'rossler':np.arange(0, 50, 0.05)}

# def model_states(model):
#     if model.__name__ in ('lorenz', 'rossler'):
#         # 时间参数
#         t = time[model.__name__]
#         # 生成模型需要的数据
#         if model.__name__ == 'lorenz':        
#             states = odeint(lorenz, (0.1, 0.1, 0.1), t, args = (10, 28, 8/3))
#         elif model.__name__ == 'rossler':
#             states = odeint(rossler, (1, 1, 1), t, args = (0.5, 2.0, 4.0))
#         # 对数据进行处理
#         states = (states - np.mean(states,0)) / np.mean((states - np.mean(states,0))**2,0)**(1/2)
#     else:
#         raise ValueError('Now noly \'lorenz\' model and \'rossler\' model are supported!')

#     return states

from scipy.integrate import odeint
import numpy as np
import json

class Model:
    def __init__(self, name):
        if name not in ('lorenz', 'rossler'):
            raise ValueError('Now noly \'lorenz\' model and \'rossler\' model are supported!')
        else:
            self.__time ={
                'lorenz' :{
                    'T0' : 0,
                    'T' : 50,
                    'delta_t' : 0.05
                },
                'rossler' :{
                    'T0' : 0,
                    'T' : 260,
                    'delta_t' : 0.1
                }
            }
            model_list = {
                'lorenz' : {
                    'init_value' : (0.1, 0.1, 0.1),
                    't' : np.arange(self.__time[name]['T0'], self.__time[name]['T'], self.__time[name]['delta_t']),
                    'args' : (10, 28, 8/3),
                    'func' : self.__lorenz,
                    'N' : 400,
                    'rho' : 1.0,
                    'sigma': 1.0,
                    'bias' : 1.0,
                    'alpha' : 1.0,
                    'T0' : self.__time[name]['T0'],
                    'T' : self.__time[name]['T'],
                    'delta_t' : self.__time[name]['delta_t']
                },
                'rossler' : {
                    'init_value' : (1.0, 1.0, 1.0),
                    't' : np.arange(self.__time[name]['T0'], self.__time[name]['T'], self.__time[name]['delta_t']),
                    'args' : (0.5, 2.0, 4.0),
                    'func' : self.__rossler,
                    'N' : 400,
                    'rho' : 1.0,
                    'sigma': 1.0,
                    'bias' : 1.0,
                    'alpha' : 1.0,
                    'T0' : self.__time[name]['T0'],
                    'T' : self.__time[name]['T'],
                    'delta_t' : self.__time[name]['delta_t']
                }
            }
            self.name = name
            self.model = model_list[name]
            states = odeint(self.model['func'], self.model['init_value'], self.model['t'], args = self.model['args'])
            states = (states - np.mean(states,0)) / np.mean((states - np.mean(states,0))**2,0)**(1/2)
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
        dx = a*(y - x)
        dy = x*(b - z) - y
        dz = x*y - c*z
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
        dy = x + a*y
        dz = b + z*(x-c)
        return np.array([dx, dy, dz]) 

    def __str__(self):
        name = self.name
        t = self.__time
        model_list = {
                name : {
                    'init_value' : str(self.model['init_value']),
                    't' : 'np.arange(%s, %s ,%s)' % (t[name]['T0'], t[name]['T'], t[name]['delta_t']),
                    'args' : str(self.model['args']),
                    'func' : '__%s in class Model' % name,
                    'N' : str(self.model['N']),
                    'rho' : str(self.model['rho']),
                    'sigma': str(self.model['sigma']),
                    'bias' : str(self.model['bias']),
                    'alpha' : str(self.model['alpha']),
                    'T0' : str(self.model['T0']),
                    'T' : str(self.model['T']),
                    'delta_t' : str(self.model['delta_t'])
                }
        }
        jsonDumpsIndentStr = json.dumps(model_list, indent = 2)
        return jsonDumpsIndentStr
    
    __repr__ = __str__
