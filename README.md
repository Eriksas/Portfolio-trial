# 投资组合优化与有效前沿分析

## 项目背景

在金融市场中，投资组合的管理与优化是至关重要的。现代投资组合理论（MPT）通过数学模型分析不同资产的组合，帮助投资者选择最优的资产配置，从而在给定风险水平下最大化收益，或在目标收益率下最小化风险。

> **重点**：本项目通过以下几个主要步骤实现投资组合优化与有效前沿的构建：

1. **数据处理与分析**：从股票数据中提取年化收益率、波动率、协方差矩阵等统计量。
2. **随机投资组合生成**：通过随机生成资产权重，计算不同组合的预期收益和风险。
3. **有效前沿的构建**：根据给定的目标收益率，通过优化算法（最小化波动率）构建有效前沿。
4. **夏普比率最大化**：计算夏普比率并找到夏普比率最大的投资组合。

## 项目功能

1. **读取股票数据**：使用`pandas`库读取并处理股票价格数据，计算对数收益率。
2. **随机投资组合生成**：通过随机生成权重并计算每个组合的年化收益率和波动率，绘制散点图。
3. **有效前沿与优化**：利用`scipy.optimize.minimize`方法，通过最小化波动率目标函数，生成有效前沿，并计算夏普比率最大化的投资组合。

## 如何使用

1. **克隆项目**：
   ```bash
   git clone https://github.com/Eriksas/Portfolio-trial.git
2. **安装依赖**：
   ```bash
   pip install numpy pandas matplotlib scipy
  准备股票数据：确保你有一个包含股票价格的Excel文件，格式如下：
  第一列为日期
  后续列为每只股票的历史价格数据

## 代码说明

1. **数据处理**
首先，我们从 Excel 文件中读取股票数据，并计算每只股票的年化平均收益率、波动率和协方差矩阵：
    ```bash
    R = np.log(data/data.shift(1))  # 按照对数收益率公式计算对数收益率
    R_mean = R.mean()*252   # 计算股票的年化平均收益率
    R_cov = R.cov()*252  # 计算协方差矩阵并年化处理
2. **随机生成投资组合**
我们通过随机生成权重，计算每个投资组合的年化收益率和波动率，并绘制散点图：
    ```bash
    portfolio_returns = []
    portfolio_volatilities = []
    num_portfolios = 10000
    for _ in range(num_portfolios):
        weights = np.random.random(5)  # 随机生成权重
        weights /= np.sum(weights)  # 权重归一化
        portfolio_return = np.sum(weights * R_mean)
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(R_cov, weights.T)))
        portfolio_returns.append(portfolio_return)
        portfolio_volatilities.append(portfolio_volatility)
4. **有效前沿优化**
我们利用 `scipy.optimize.minimize` 方法，通过最小化波动率的目标函数，构建有效前沿：
    ```bash
     def minimize_volatility(weights):    # 最小化波动率目标函数
     return portfolio_statistics(weights)[1]

     constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
     target_returns = np.linspace(R_mean.min(), R_mean.max(), 100)
     efficient_volatilities = []
     for target in target_returns:
       constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x * R_mean) - target})
       result = minimize(minimize_volatility, num_assets * [1. / num_assets],
                      bounds=bounds, constraints=constraints)
      efficient_volatilities.append(result.fun)
      constraints.pop()  # 移除目标收益率约束
4. **夏普比率最大化**
最后，我们计算夏普比率并标记夏普比率最大化的点：
   ```bash
   sharpe_ratios = (np.array(target_returns) - risk_free_rate) / np.array(efficient_volatilities)
   max_sharpe_idx = np.argmax(sharpe_ratios)  # 找到夏普比率最大的索引
   max_sharpe_return = target_returns[max_sharpe_idx]
   max_sharpe_volatility = efficient_volatilities[max_sharpe_idx]
5. **绘制结果图形**
通过`matplotlib`，我们绘制投资组合的有效前沿图，标记出夏普比率最大化的点和波动率最小的点。
    ```bash
      plt.scatter(portfolio_volatilities, portfolio_returns,
            c='blue', alpha=0.3, s=15, label='随机投资组合', edgecolor='k')

      plt.plot(efficient_volatilities, target_returns,
         c='red', linewidth=2.5, linestyle='-', label='有效前沿')

      plt.scatter(max_sharpe_volatility, max_sharpe_return,
            c='gold', s=100, edgecolors='black', label='夏普比率最大点')

      plt.scatter(min_volatility, min_volatility_return,
            c='green', s=100, edgecolors='black', label='波动率最小点')
输出示例
有效前沿图：展示了不同投资组合的年化收益率与波动率之间的关系，并标注了波动率最小点和夏普比率最大点。
组合标准差与收益率关系：分析了不同相关系数下，两个资产组合的风险与收益。
注意：当你在运行代码时，输出图形将自动显示在屏幕上，并包括相关的投资组合优化结果。

> **学习参考来源**：[https://zhuanlan.zhihu.com/p/344624949](url)

许可证
该项目遵循 MIT 许可证。
