import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numpy as np
import math
from numba import jit
import time

fontdict = {'family': 'Times New Roman' ,'size': 10}
@jit(cache=True, nopython=True)
def FHN(X, t, a, b, c, f):
    (x, y) = X
    dx = x * (x - 1)*(1 - b * x) - y + a / \
              (2 * np.pi * f) * np.cos(2 * f*np.pi * t)
    dy = c * x

    return [dx, dy]

dt = 0.01
t = np.arange(0, 400, dt)
df = 0.000005
time_start = time.time()
for f in np.arange(0.12, 0.14, df):
    res = []
    arg = (0.1, 10, 1.0, f)
    states = np.array(odeint(FHN, (0, 1.0), t, args=arg))
    S = states[:,1]
    for i in range(1, len(S) - 1):
        if S[i] >= 0 and (S[i + 1] - S[i]) * (S[i] - S[i - 1]) < 0:
            res.append(i * dt)
    # print(res)
    data = []
    # f = 0.129;
    for i in range(1, len(res)):
        data.append([f, res[i] - res[i - 1]])
    data = np.array(data)
    plt.plot(data[:, 0], data[:, 1], '.',color='#003399', markersize=0.8)

time_end = time.time()
print("df = %s" % df)
print("total time: %ss" %(time_end - time_start))

plt.xlabel('f', fontdict=fontdict)
plt.ylabel('inter-spike interval (ISI)', fontdict=fontdict)
# plt.title()
plt.show()


# dt = 0.001
# total time: 1.4371569156646729s
# dt = 0.0001
# total time: 9.782833099365234s
# dt = 0.00001
# total time: 92.912917137146s
# dt = 0.000001
# total time: 1306.3604617118835s
