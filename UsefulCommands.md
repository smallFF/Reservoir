# 相关的MATLAB命令

## 图片相关
- `saveas(gcf, 'file.png')` - 当前图窗保存为`file.png`文件

- `print(gcf, '-dpng','-r600', 'file.png')` - 当前图窗保存为 `dpi = 600` 的 `file.png` 文件

- `suptitle(str)` - 可以给同一图窗的多个子图加一个总标题

## 计算相关
- `[~, y] = ode45(@(t,X) lorenz(t,X, args), t, init_value)` - 求解微分方程的过程中可以传入参数

- `opt.disp = 0; rhoW = abs(eigs(W, 1, 'LM', opt))` - 可返回矩阵 `W` 最大特征值的模，用于计算谱半径

# 相关的Python命令

## 图片相关
- `plt.savefig('file.png', dpi = 600)` - 把图窗保存为 `dpi = 600` 的 `file.png` 图片

- `plt.gcf().canvas.get_supported_filetypes()` - 返回 `savefig` 函数支持保存的图片格式

- `plt.plot(x, y,ls = '-', linewidth = 1)` - 设置线的格式：类型 = 实线，线宽 = 1

## 计算相关
- `states = odeint(lorenz, init_value, t, args = args)` - 求解微分方程的过程中可传入参数

- `rho = max(np.abs(np.linalg.eig(W)[0])` - 可返回矩阵 `W` 最大特征值的模，用于计算谱半径