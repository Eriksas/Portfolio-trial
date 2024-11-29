import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import pandas as pd
from scipy.optimize import minimize

# 设置中文字体和负号显示
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 构建投资组合
data = pd.read_excel('E:/work/Stock Portfolio.xlsx', header=0, index_col=0)
data.head()
(data/data.iloc[0]).plot(figsize=(8, 6))

R = np.log(data/data.shift(1))  # 按照对数收益率公式计算对数收益率
R = R.dropna()  # 删除缺省的数据
R_mean = R.mean()*252   # 计算股票的年化平均收益率,一年有效交易日为252天
print('各只股票的收益率均值:''\n', R_mean)

R_cov = R.cov()*252  # 计算协方差矩阵并年化处理
print('计算协方差矩阵并年化处理：''\n', R_cov)

R_corr = R.corr()    # 计算相关系数矩阵
print('计算相关系数矩阵：''\n', R_corr)

R_vol = R.std()*np.sqrt(252)  # 计算股票年化收益率的波动率
print('计算股票年化收益率的波动率:''\n', R_vol)

X = np.random.random(5)
weights = X / np.sum(X)  # 生成和为1的随机数权重矩阵
print(weights)

R_port = np.sum(weights * R_mean)
print('投资组合的预期收益率:', round(R_port, 4))

volatility_port = np.sqrt(np.dot(weights, np.dot(R_cov, weights.T)))
print('投资组合收益率波动率', round(volatility_port, 4))

# 投资组合收益率和波动率的列表
portfolio_returns = []
portfolio_volatilities = []

# 生成随机投资组合
num_portfolios = 10000
for _ in range(num_portfolios):
    weights = np.random.random(5)  # 随机生成权重
    weights /= np.sum(weights)  # 权重归一化
    # 计算组合收益率和波动率
    portfolio_return = np.sum(weights * R_mean)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(R_cov, weights.T)))
    # 添加到结果列表
    portfolio_returns.append(portfolio_return)
    portfolio_volatilities.append(portfolio_volatility)


# 投资组合优化函数
def portfolio_statistics(weights):    # 输入权重，计算投资组合的年化收益率和波动率
    portfolio_return = np.sum(weights * R_mean)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(R_cov, weights)))
    return portfolio_return, portfolio_volatility


def minimize_volatility(weights):    # 最小化波动率目标函数
    return portfolio_statistics(weights)[1]


# 定义边界和约束
num_assets = len(R_mean)
bounds = [(0, 1) for _ in range(num_assets)]  # 权重范围在0到1之间
constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # 权重之和为1

# 构造有效前沿
target_returns = np.linspace(R_mean.min(), R_mean.max(), 100)
efficient_volatilities = []

for target in target_returns:
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * R_mean) - target})
    # 目标收益率约束
    result = minimize(minimize_volatility, num_assets * [1. / num_assets],
                      bounds=bounds, constraints=constraints)
    efficient_volatilities.append(result.fun)
    constraints.pop()  # 移除目标收益率约束，准备下一个优化

# 计算无风险收益率（假设为3%）
risk_free_rate = 0.03

# 计算夏普比率最大化的投资组合点
sharpe_ratios = (np.array(target_returns) - risk_free_rate) / np.array(efficient_volatilities)
max_sharpe_idx = np.argmax(sharpe_ratios)  # 找到夏普比率最大的索引
max_sharpe_return = target_returns[max_sharpe_idx]
max_sharpe_volatility = efficient_volatilities[max_sharpe_idx]

plt.figure(figsize=(13, 9))  # 调整画布大小

# 随机组合散点图
plt.scatter(portfolio_volatilities, portfolio_returns,
            c='blue', alpha=0.3, s=15, label='随机投资组合', edgecolor='k')

# 有效前沿曲线
plt.plot(efficient_volatilities, target_returns,
         c='red', linewidth=2.5, linestyle='-', label='有效前沿')

# 标记夏普比率最大化点
plt.scatter(max_sharpe_volatility, max_sharpe_return,
            c='gold', s=100, edgecolors='black', label='夏普比率最大点')
plt.text(max_sharpe_volatility, max_sharpe_return,
         f'  ({max_sharpe_volatility:.2f}, {max_sharpe_return:.2f})',
         fontsize=12, color='black')
# 计算波动率最小的点
min_volatility_idx = np.argmin(np.array(portfolio_volatilities))  # 找到波动率最小的索引
min_volatility_return = portfolio_returns[min_volatility_idx]
min_volatility = portfolio_volatilities[min_volatility_idx]

# 标记波动率最小的点
plt.scatter(min_volatility, min_volatility_return,
            c='green', s=100, edgecolors='black', label='波动率最小点')
plt.text(min_volatility, min_volatility_return,
         f'({min_volatility:.2f}, {min_volatility_return:.2f})',
         fontsize=12, color='gray')

print('波动率波动率在可行性集是全局最小值时的投资组合的预期收益率',
      round(min_volatility_return, 4))
print('在可行集是全局最小的波动率', round(min_volatility, 4))

plt.xlabel('波动率 (年化)', fontsize=16)
plt.ylabel('收益率 (年化)', fontsize=16, rotation=0, labelpad=20)
plt.title('投资组合的有效前沿（标注夏普比率最大化点）', fontsize=20, pad=20)
plt.title('投资组合的有效前沿', fontsize=20, pad=20)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
# 添加图例
plt.legend(fontsize=14, loc='upper left', frameon=True, shadow=True)
plt.tight_layout()
plt.show()


# 定义参数，寻找两种不同资产的组合标准差和收益率关系
sd_1 = 0.25  # 资产1的标准差
sd_2 = 0.11  # 资产2的标准差
r1 = 0.17    # 资产1的预期收益率
r2 = 0.06    # 资产2的预期收益率

# 资产1和资产2的权重
w1 = np.linspace(0, 1, 11)
w2 = 1 - w1

# 计算组合收益率
rp = r1 * w1 + r2 * w2

# 相关系数数组
r_values = [0, 0.5, 1, -0.5, -1]

# 初始化存储标准差的矩阵
sd_p = np.zeros((11, 5))

# 计算不同相关系数下的组合标准差
for j, r in enumerate(r_values):
    sd_p[:, j] = np.sqrt((w1 * sd_1)**2 + (w2 * sd_2)**2
                         + 2 * w1 * w2 * r * sd_1 * sd_2)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制每个相关系数下的组合标准差与收益率关系
colors = ['red', 'yellow', 'green', 'blue', 'magenta']
labels = [f'相关系数r={r}' for r in r_values]

for j, color in enumerate(colors):
    plt.scatter(sd_p[:, j], rp, color=color, label=labels[j], s=30)  # 绘制散点
    plt.plot(sd_p[:, j], rp, color=color, lw=2)  # 绘制折线

# 设置图表标题和标签
plt.title(u'不同相关系数下，2种投资组合的风险与收益', fontsize=18)
plt.xlabel(u'组合标准差', fontsize=14)
plt.ylabel(u'收益率', fontsize=14, rotation=0)

# 设置坐标轴的刻度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 添加图例
plt.legend(fontsize=12)

# 添加网格
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
