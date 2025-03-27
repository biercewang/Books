import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS 系统可用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 尝试从多个数据源获取数据
def get_stock_data(ticker='AAPL', start_date=None, end_date=None):
    """尝试从多个数据源获取股票数据"""
    data_sources = ['stooq', 'yahoo']
    
    for source in data_sources:
        try:
            print(f"尝试从 {source} 获取 {ticker} 数据...")
            data = web.DataReader(ticker, source, start_date, end_date)
            if not data.empty:
                print(f"成功从 {source} 获取数据")
                return data
        except Exception as e:
            print(f"从 {source} 获取数据失败: {e}")
    
    raise Exception("无法从任何数据源获取数据")

# 主程序
try:
    # 设置时间范围
    start_date = datetime.datetime(2020, 1, 1)  # 扩大时间范围
    end_date = datetime.datetime(2024, 1, 1)
    
    # 尝试获取不同股票的数据
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    
    for ticker in tickers:
        try:
            data = get_stock_data(ticker, start_date, end_date)
            print(f"成功获取 {ticker} 数据")
            break
        except Exception as e:
            print(f"{ticker} 数据获取失败: {e}")
    
    if 'data' not in locals() or data.empty:
        raise Exception("无法获取任何股票数据")
        
    # 计算对数收益率
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # 计算统计参数
    returns = data['Log Return'].dropna()
    mean = returns.mean()
    std = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()
    
    # 绘制核密度估计图
    plt.figure(figsize=(10, 6))
    sns.kdeplot(returns, fill=True, color='skyblue')
    
    # 添加统计参数文本
    stats_text = f'均值: {mean:.4f}\n标准差: {std:.4f}\n偏度: {skew:.4f}\n峰度: {kurt:.4f}'
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f'{ticker} 日收益率的核密度估计')
    plt.xlabel('对数收益率')
    plt.ylabel('密度')
    plt.grid(True)
    plt.show()
    
except Exception as e:
    print(f"错误: {e}")
